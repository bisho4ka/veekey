#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <corecrt_math_defines.h>
#include <cmath>
#include <veekay/veekay.hpp>
#include <vulkan/vulkan_core.h>
#include <imgui.h>

namespace {
    constexpr uint32_t max_models = 1024;
    constexpr uint32_t max_point_lights = 16;
    size_t aligned_sizeof;

    float mouse_sens = 0.75f;
    bool camera_control = false;

    struct DirectionalLight {
        veekay::vec3 color = {1.0f, 1.0f, 1.0f};
        float intensity = 0.2f;
        veekay::vec3 direction = {0.15f, 1.0f, 0.3f};
    };
    DirectionalLight directional_light;

    struct Vertex {
        veekay::vec3 position;
        veekay::vec3 normal;
        veekay::vec2 uv;
    };

    // scene data structure
    struct SceneUniforms {
        veekay::mat4 view_projection;
        veekay::vec3 view_position;
        float _pad0;
        veekay::vec3 ambient_light_intensity;
        float _pad1;
        veekay::vec3 sun_light_direction;
        float _pad2;
        veekay::vec3 sun_light_color;
        float _pad3;
        uint32_t point_lights_count;
        uint32_t _pad4[3];
    };

    // model data structure with material properties
    struct ModelUniforms {
        veekay::mat4 model;           // model matrix
        veekay::vec3 albedo_color;     // diffuse component
        float _pad0;
        veekay::vec3 specular_color;   // specular color
        float _pad2;
        float shininess;               // shininess parameter
        uint32_t _pad3[3];
    };

    typedef struct PointLight {
        veekay::vec3 position;
        float radius;                   // for inverse square law attenuation
        veekay::vec3 color;
        float _pad0;
    } PointLight;

    // 3D geometry
    struct Mesh {
        veekay::graphics::Buffer* vertex_buffer;
        veekay::graphics::Buffer* index_buffer;
        uint32_t indices;
    };

    struct Transform {
        veekay::vec3 position = {};
        veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
        veekay::vec3 rotation = {};

        veekay::mat4 matrix() const;
    };

    // Model with material properties
    struct Model {
        Mesh mesh;
        Transform transform;

        veekay::vec3 albedo_color;      // diffuse color
        veekay::vec3 specular_color;    // specular color
        float shininess;                 // shininess for Blinn-Phong
    };

    struct Camera {
        constexpr static float default_fov = 60.0f;
        constexpr static float default_near_plane = 0.01f;
        constexpr static float default_far_plane = 100.0f;

        veekay::vec3 position = {};
        veekay::vec3 rotation = {};

        float fov = default_fov;
        float near_plane = default_near_plane;
        float far_plane = default_far_plane;

        veekay::mat4 view() const;                     // Look-At matrix
        veekay::mat4 view_projection(float aspect_ratio) const;
        veekay::vec3 forward() const;
    };

    // Scene objects
    inline namespace {
        Camera camera {
            .position = {-0.3f, -2.0f, -3.6f},
            .rotation = {0.32f, 0.2f, 0.0f}
        };

        std::vector<Model> models;
        std::vector<PointLight> point_lights;
    }

    // Vulkan objects
    inline namespace {
        VkShaderModule vertex_shader_module;
        VkShaderModule fragment_shader_module;

        VkDescriptorPool descriptor_pool;
        VkDescriptorSetLayout descriptor_set_layout;
        VkDescriptorSet descriptor_set;

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        veekay::graphics::Buffer* scene_uniforms_buffer;
        veekay::graphics::Buffer* model_uniforms_buffer;
        veekay::graphics::Buffer* point_lights_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;
    }

    float toRadians(float degrees) {
        return degrees * float(M_PI) / 180.0f;
    }

    veekay::mat4 Transform::matrix() const {
        veekay::mat4 scale_mat = veekay::mat4::scaling(scale);
        veekay::mat4 y_rot = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
        veekay::mat4 x_rot = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
        veekay::mat4 z_rot = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
        veekay::mat4 rotation_mat = y_rot * x_rot * z_rot;
        veekay::mat4 translation_mat = veekay::mat4::translation(position);
        return translation_mat * rotation_mat * scale_mat;
    }

    // Look-At matrix implementation
    veekay::mat4 look_at_matrix(const veekay::vec3& eye, const veekay::vec3& target, const veekay::vec3& up) {
        veekay::vec3 z = veekay::vec3::normalized(eye - target);
        veekay::vec3 x = veekay::vec3::normalized(veekay::vec3::cross(up, z));
        veekay::vec3 y = veekay::vec3::cross(z, x);

        veekay::mat4 result;
        result[0][0] = x.x; result[0][1] = y.x; result[0][2] = z.x; result[0][3] = 0.0f;
        result[1][0] = x.y; result[1][1] = y.y; result[1][2] = z.y; result[1][3] = 0.0f;
        result[2][0] = x.z; result[2][1] = y.z; result[2][2] = z.z; result[2][3] = 0.0f;
        result[3][0] = -veekay::vec3::dot(x, eye);
        result[3][1] = -veekay::vec3::dot(y, eye);
        result[3][2] = -veekay::vec3::dot(z, eye);
        result[3][3] = 1.0f;

        return result;
    }

    veekay::vec3 Camera::forward() const {
        float y_rad = rotation.y;
        float x_rad = rotation.x;

        float cos_y = cosf(y_rad);
        float sin_y = sinf(y_rad);
        float cos_x = -cosf(x_rad);
        float sin_x = -sinf(x_rad);

        veekay::vec3 result = {sin_y * cos_x, sin_x, cos_y * cos_x};
        return veekay::vec3::normalized(result);
    }

    // View matrix using Look-At
    veekay::mat4 Camera::view() const {
        veekay::vec3 forward = Camera::forward();
        veekay::vec3 target = position + forward;
        veekay::vec3 up = {0.0f, 1.0f, 0.0f};
        return look_at_matrix(position, target, up);
    }

    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
        return view() * projection;
    }

    VkShaderModule loadShaderModule(const char* path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size,
                .pCode = buffer.data(),
        };

        VkShaderModule result;
        vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result);
        return result;
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_device, &props);
        uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
        aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;

        // Load shaders
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");

        // Pipeline creation (simplified)
        VkPipelineShaderStageCreateInfo stage_infos[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vertex_shader_module,
                .pName = "main",
            },
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = fragment_shader_module,
                .pName = "main",
            }
        };

        // Vertex input with normals
        VkVertexInputBindingDescription buffer_binding{
            .binding = 0,
            .stride = sizeof(Vertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };

        VkVertexInputAttributeDescription attributes[] = {
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},  // normal attribute
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
        };

        VkPipelineVertexInputStateCreateInfo input_state_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &buffer_binding,
            .vertexAttributeDescriptionCount = 3,
            .pVertexAttributeDescriptions = attributes,
        };

        // Descriptor pool
        VkDescriptorPoolSize pools[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}
        };

        VkDescriptorPoolCreateInfo pool_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 3,
            .pPoolSizes = pools,
        };
        vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool);

        // Descriptor set layout
        VkDescriptorSetLayoutBinding bindings[] = {
            {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
            {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, nullptr},
            {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        };

        VkDescriptorSetLayoutCreateInfo layout_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 3,
            .pBindings = bindings,
        };
        vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_set_layout);

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };
        vkAllocateDescriptorSets(device, &alloc_info, &descriptor_set);

        // Pipeline layout
        VkPipelineLayoutCreateInfo pipeline_layout_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };
        vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout);

        // Create buffers
        scene_uniforms_buffer = new veekay::graphics::Buffer(
            sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
            max_models * aligned_sizeof, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        point_lights_buffer = new veekay::graphics::Buffer(
            max_point_lights * sizeof(PointLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // Update descriptor set
        VkDescriptorBufferInfo buffer_infos[] = {
            {scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms)},
            {model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms)},
            {point_lights_buffer->buffer, 0, max_point_lights * sizeof(PointLight)},
        };

        VkWriteDescriptorSet write_infos[] = {
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buffer_infos[0], nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &buffer_infos[1], nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[2], nullptr},
        };
        vkUpdateDescriptorSets(device, 3, write_infos, 0, nullptr);

        // Create meshes with normals
        // Plane mesh
        std::vector<Vertex> plane_vertices = {
            {{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
            {{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
            {{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
        };
        std::vector<uint32_t> plane_indices = {0, 1, 2, 2, 3, 0};

        plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
            plane_vertices.size() * sizeof(Vertex), plane_vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        plane_mesh.index_buffer = new veekay::graphics::Buffer(
            plane_indices.size() * sizeof(uint32_t), plane_indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        plane_mesh.indices = uint32_t(plane_indices.size());

        // Cube mesh with normals
        std::vector<Vertex> cube_vertices = {
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
            {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
        };

        std::vector<uint32_t> cube_indices = {
            0,1,2, 2,3,0, 4,5,6, 6,7,4, 8,9,10, 10,11,8,
            12,13,14, 14,15,12, 16,17,18, 18,19,16, 20,21,22, 22,23,20
        };

        cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
            cube_vertices.size() * sizeof(Vertex), cube_vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        cube_mesh.index_buffer = new veekay::graphics::Buffer(
            cube_indices.size() * sizeof(uint32_t), cube_indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        cube_mesh.indices = uint32_t(cube_indices.size());

        // Add models with material properties
        models.emplace_back(Model{plane_mesh, Transform{}, {0.98f, 0.85f, 0.90f}, {0.94f, 0.90f, 0.92f}, 32.0f});
        models.emplace_back(Model{cube_mesh, Transform{{2.0f, -0.5f, -1.0f}, {0.8f,0.8f,0.8f}, {0.0f, M_PI/6.0f, 0.0f}}, {0.95f,0.88f,0.75f}, {0.90f,0.92f,0.95f}, 128.0f});
        models.emplace_back(Model{cube_mesh, Transform{{-2.0f, -0.5f, -0.5f}, {0.9f,0.9f,0.9f}, {0.0f, M_PI/4.0f, 0.0f}}, {0.88f,0.95f,0.85f}, {0.92f,0.95f,0.90f}, 96.0f});
        models.emplace_back(Model{cube_mesh, Transform{{0.0f, -0.5f, 1.0f}, {1.0f,1.0f,1.0f}, {0.0f,0.0f,0.0f}}, {0.92f,0.85f,0.95f}, {0.98f,0.94f,0.90f}, 156.0f});

        // Add point lights
        point_lights.emplace_back(PointLight{{2.5f, -1.2f, -0.45f}, 4.0f, {1.0f, 0.95f, 0.85f}});
        point_lights.emplace_back(PointLight{{0.0f, -0.5f, -0.2f}, 4.0f, {1.0f, 0.90f, 0.80f}});
        point_lights.emplace_back(PointLight{{-2.2f, -1.1f, -0.25f}, 4.0f, {0.95f, 0.90f, 1.0f}});
    }

    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;
        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;
        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;
        delete point_lights_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        // Minimal UI for controlling light parameters as required
        ImGui::Begin("Light Controls");

        ImGui::Text("Ambient Light");
        static float ambient_intensity = 0.025f;
        ImGui::SliderFloat("Intensity", &ambient_intensity, 0.0f, 1.0f);

        ImGui::Separator();
        ImGui::Text("Directional Light");
        ImGui::SliderFloat("Dir Intensity", &directional_light.intensity, 0.0f, 2.0f);
        ImGui::ColorEdit3("Dir Color", &directional_light.color.x);
        ImGui::SliderFloat3("Dir Direction", &directional_light.direction.x, -1.0f, 1.0f);
        directional_light.direction = veekay::vec3::normalized(directional_light.direction);

        ImGui::Separator();
        ImGui::Text("Point Lights (%zu)", point_lights.size());
        if (ImGui::Button("Add Point Light")) {
            if (point_lights.size() < max_point_lights) {
                point_lights.push_back(PointLight{camera.position, 5.0f, {1.0f, 0.9f, 0.95f}});
            }
        }

        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(i);
            if (ImGui::CollapsingHeader(("Light " + std::to_string(i)).c_str())) {
                ImGui::ColorEdit3("Color", &point_lights[i].color.x);
                ImGui::SliderFloat3("Pos", &point_lights[i].position.x, -10.0f, 10.0f);
                ImGui::SliderFloat("Radius", &point_lights[i].radius, 0.1f, 20.0f);
                if (ImGui::Button("Remove")) {
                    point_lights.erase(point_lights.begin() + i);
                    ImGui::PopID();
                    break;
                }
            }
            ImGui::PopID();
        }

        ImGui::End();

        // Camera control with keyboard and mouse only (no UI)
        {
            using namespace veekay::input;
            if (camera_control) {
                auto move_delta = mouse::cursorDelta();
                camera.rotation.y = fmod(2.0f * M_PI + camera.rotation.y + move_delta.x * mouse_sens * 0.02f, 2.0f * M_PI);
                camera.rotation.x = std::clamp(camera.rotation.x + move_delta.y * mouse_sens * 0.02f, -1.5f, 1.5f);
            }

            veekay::vec3 front = {sinf(camera.rotation.y) * cosf(camera.rotation.x), 
                                  sinf(camera.rotation.x), 
                                  cosf(camera.rotation.y) * cosf(camera.rotation.x)};
            front = veekay::vec3::normalized(front);
            veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, {0.0f, 1.0f, 0.0f}));
            veekay::vec3 up = {0.0f, 1.0f, 0.0f};

            if (keyboard::isKeyPressed(keyboard::Key::c)) {
                camera_control = !camera_control;
                mouse::setCaptured(camera_control);
            }

            float speed = 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::w)) camera.position += front * speed;
            if (keyboard::isKeyDown(keyboard::Key::s)) camera.position -= front * speed;
            if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * speed;
            if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * speed;
            if (keyboard::isKeyDown(keyboard::Key::q)) camera.position += up * speed;
            if (keyboard::isKeyDown(keyboard::Key::z)) camera.position -= up * speed;
        }

        // Update uniforms
        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio),
            .view_position = camera.position,
            .ambient_light_intensity = {ambient_intensity, ambient_intensity, ambient_intensity},
            .sun_light_direction = directional_light.direction,
            .sun_light_color = directional_light.color * directional_light.intensity,
            .point_lights_count = static_cast<uint32_t>(point_lights.size()),
        };

        std::vector<ModelUniforms> model_uniforms(models.size());
        for (size_t i = 0; i < models.size(); ++i) {
            const Model& model = models[i];
            model_uniforms[i] = {
                .model = model.transform.matrix(),
                .albedo_color = model.albedo_color,
                .specular_color = model.specular_color,
                .shininess = model.shininess
            };
        }

        memcpy(scene_uniforms_buffer->mapped_region, &scene_uniforms, sizeof(SceneUniforms));

        uint8_t* base = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);
        for (size_t i = 0; i < model_uniforms.size(); ++i) {
            memcpy(base + i * aligned_sizeof, &model_uniforms[i], sizeof(ModelUniforms));
        }

        if (!point_lights.empty()) {
            memcpy(point_lights_buffer->mapped_region, point_lights.data(), 
                   point_lights.size() * sizeof(PointLight));
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &begin_info);

        VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
        VkClearValue clear_depth{.depthStencil = {1.0f, 0}};
        VkClearValue clear_values[] = {clear_color, clear_depth};

        VkRenderPassBeginInfo render_pass_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = veekay::app.vk_render_pass,
            .framebuffer = framebuffer,
            .renderArea = {.offset = {0,0}, .extent = {veekay::app.window_width, veekay::app.window_height}},
            .clearValueCount = 2,
            .pClearValues = clear_values,
        };
        vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize offset = 0;

        VkBuffer current_vb = VK_NULL_HANDLE;
        VkBuffer current_ib = VK_NULL_HANDLE;

        for (size_t i = 0; i < models.size(); ++i) {
            const Mesh& mesh = models[i].mesh;

            if (current_vb != mesh.vertex_buffer->buffer) {
                current_vb = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vb, &offset);
            }

            if (current_ib != mesh.index_buffer->buffer) {
                current_ib = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_ib, 0, VK_INDEX_TYPE_UINT32);
            }

            uint32_t dynamic_offset = uint32_t(i * aligned_sizeof);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                   0, 1, &descriptor_set, 1, &dynamic_offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }
}

int main() {
    return veekay::run({initialize, shutdown, update, render});
}