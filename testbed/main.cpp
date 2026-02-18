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
#include <lodepng.h>  // добавляем для загрузки PNG

#ifndef M_PI
constexpr float M_PI = 3.14159265358979323846f;
#endif

namespace {
    constexpr uint32_t max_models = 1024;
    constexpr uint32_t max_point_lights = 16;
    constexpr uint32_t max_materials = 8;  // максимум материалов
    size_t aligned_sizeof;

    float mouse_sens = 0.75f;
    float ambient_light = 0.025f;
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
        veekay::vec2 uv;  // текстурные координаты
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

    // model data structure
    struct ModelUniforms {
        veekay::mat4 model;
        veekay::vec3 albedo_color;
        float _pad0;
        veekay::vec3 specular_color;
        float _pad2;
        float shininess;
        uint32_t _pad3[3];
    };

    typedef struct PointLight {
        veekay::vec3 position;
        float radius;
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

    // all components in one
    struct Model {
        Mesh mesh;
        Transform transform;
        veekay::vec3 albedo_color;
        veekay::vec3 specular_color;
        float shininess;
        int material_id = 0;  // индекс материала
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

        veekay::mat4 view() const;
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
        
        // Два layout'а: для сцены и для материалов
        VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout material_set_layout = VK_NULL_HANDLE;
        
        VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> material_descriptor_sets;  // наборы для материалов

        VkPipelineLayout pipeline_layout;
        VkPipeline pipeline;

        veekay::graphics::Buffer* scene_uniforms_buffer;
        veekay::graphics::Buffer* model_uniforms_buffer;
        veekay::graphics::Buffer* point_lights_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;

        // Текстурные ресурсы
        veekay::graphics::Texture* missing_texture;
        VkSampler missing_texture_sampler;
        VkSampler material_sampler = VK_NULL_HANDLE;

        // Материалы
        struct Material {
            veekay::graphics::Texture* albedo = nullptr;
            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        };
        std::vector<Material> materials;
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

    veekay::mat4 look_at_matrix(const veekay::vec3& eye_vtr,
                                const veekay::vec3& target,
                                const veekay::vec3& world_y) {
        veekay::vec3 axis_z = veekay::vec3::normalized(eye_vtr - target);
        veekay::vec3 axis_x = veekay::vec3::normalized(veekay::vec3::cross(world_y, axis_z));
        veekay::vec3 axis_y = veekay::vec3::cross(axis_z, axis_x);

        veekay::mat4 result_look_at;
        result_look_at[0][0] = axis_x.x; result_look_at[0][1] = axis_y.x; result_look_at[0][2] = axis_z.x; result_look_at[0][3] = 0.0f;
        result_look_at[1][0] = axis_x.y; result_look_at[1][1] = axis_y.y; result_look_at[1][2] = axis_z.y; result_look_at[1][3] = 0.0f;
        result_look_at[2][0] = axis_x.z; result_look_at[2][1] = axis_y.z; result_look_at[2][2] = axis_z.z; result_look_at[2][3] = 0.0f;
        result_look_at[3][0] = -veekay::vec3::dot(axis_x, eye_vtr);
        result_look_at[3][1] = -veekay::vec3::dot(axis_y, eye_vtr);
        result_look_at[3][2] = -veekay::vec3::dot(axis_z, eye_vtr);
        result_look_at[3][3] = 1.0f;

        return result_look_at;
    }

    veekay::vec3 Camera::forward() const {
        float y_rad = rotation.y;
        float x_rad = rotation.x;

        float cos_y = cosf(y_rad);
        float sin_y = sinf(y_rad);
        float cos_x = -cosf(x_rad);
        float sin_x = -sinf(x_rad);

        veekay::vec3 result_forward = {sin_y * cos_x, sin_x, cos_y * cos_x};
        result_forward = veekay::vec3::normalized(result_forward);

        return result_forward;
    }

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
        if (!file.is_open()) return VK_NULL_HANDLE;
        
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

    // Функция загрузки текстуры из PNG
    veekay::graphics::Texture* loadTextureFromPNG(VkCommandBuffer cmd, const char* path) {
        std::vector<unsigned char> image;
        unsigned width, height;

        unsigned result_decode = lodepng::decode(image, width, height, path);

        if (result_decode) {
            std::cerr << "Failed to load image (PNG) '" << path << "' : " << result_decode << "\n";
            return nullptr;
        }

        std::vector<uint32_t> pixels(width * height);

        for (size_t i = 0; i < width * height; ++i) {
            uint8_t r = image[4 * i + 0];
            uint8_t g = image[4 * i + 1];
            uint8_t b = image[4 * i + 2];
            uint8_t a = image[4 * i + 3];
            pixels[i] = (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
        }

        return new veekay::graphics::Texture(cmd, width, height, VK_FORMAT_B8G8R8A8_UNORM, pixels.data());
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_device, &props);
        uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
        aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;

        { // Build graphics pipeline
            vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
            fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");

            VkPipelineShaderStageCreateInfo stage_infos[2] = {
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_module, "main", nullptr},
                {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_module, "main", nullptr}
            };

            VkVertexInputBindingDescription buffer_binding{
                    .binding = 0,
                    .stride = sizeof(Vertex),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            VkVertexInputAttributeDescription attributes[] = {
                    {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
                    {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
                    {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},  // текстурные координаты
            };

            VkPipelineVertexInputStateCreateInfo input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = 3,
                    .pVertexAttributeDescriptions = attributes,
            };

            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            VkPipelineRasterizationStateCreateInfo raster_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    .polygonMode = VK_POLYGON_MODE_FILL,
                    .cullMode = VK_CULL_MODE_BACK_BIT,
                    .frontFace = VK_FRONT_FACE_CLOCKWISE,
                    .lineWidth = 1.0f,
            };

            VkPipelineMultisampleStateCreateInfo sample_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            };

            VkViewport viewport{
                    .x = 0.0f, .y = 0.0f,
                    .width = static_cast<float>(veekay::app.window_width),
                    .height = static_cast<float>(veekay::app.window_height),
                    .minDepth = 0.0f, .maxDepth = 1.0f,
            };

            VkRect2D scissor{
                    .offset = {0, 0},
                    .extent = {veekay::app.window_width, veekay::app.window_height},
            };

            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    .viewportCount = 1, .pViewports = &viewport,
                    .scissorCount = 1, .pScissors = &scissor,
            };

            VkPipelineDepthStencilStateCreateInfo depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true, .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            VkPipelineColorBlendAttachmentState attachment_info{
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            };

            VkPipelineColorBlendStateCreateInfo blend_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    .logicOpEnable = false,
                    .attachmentCount = 1, .pAttachments = &attachment_info
            };

            // Создаем пул дескрипторов с учетом текстур
            {
                VkDescriptorPoolSize pools[] = {
                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8},
                        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 8},
                        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_materials},  // для текстур
                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
                };

                VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = 1 + max_materials,  // 1 для сцены + max_materials для материалов
                        .poolSizeCount = 4,
                        .pPoolSizes = pools,
                };

                vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool);
            }

            // Layout для сцены (uniforms + storage buffers)
            {
                VkDescriptorSetLayoutBinding bindings[] = {
                        {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
                        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, nullptr},
                        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                };

                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = 3,
                        .pBindings = bindings,
                };

                vkCreateDescriptorSetLayout(device, &info, nullptr, &scene_set_layout);
            }

            // Layout для материалов (только текстуры)
            {
                VkDescriptorSetLayoutBinding material_bindings[] = {
                        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                };

                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = 1,
                        .pBindings = material_bindings,
                };

                vkCreateDescriptorSetLayout(device, &info, nullptr, &material_set_layout);
            }

            // Выделяем descriptor set для сцены
            {
                VkDescriptorSetAllocateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = descriptor_pool,
                        .descriptorSetCount = 1,
                        .pSetLayouts = &scene_set_layout,
                };
                vkAllocateDescriptorSets(device, &info, &scene_descriptor_set);
            }

            // Выделяем descriptor set'ы для материалов
            {
                material_descriptor_sets.resize(max_materials);
                std::vector<VkDescriptorSetLayout> material_layouts(max_materials, material_set_layout);

                VkDescriptorSetAllocateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = descriptor_pool,
                        .descriptorSetCount = (uint32_t)material_layouts.size(),
                        .pSetLayouts = material_layouts.data()
                };
                vkAllocateDescriptorSets(device, &info, material_descriptor_sets.data());
            }

            // Layout пайплайна с двумя наборами
            {
                VkDescriptorSetLayout setLayouts[] = { scene_set_layout, material_set_layout };
                VkPipelineLayoutCreateInfo layout_info{
                        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                        .setLayoutCount = 2,
                        .pSetLayouts = setLayouts,
                };
                vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
            }

            VkGraphicsPipelineCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    .stageCount = 2,
                    .pStages = stage_infos,
                    .pVertexInputState = &input_state_info,
                    .pInputAssemblyState = &assembly_state_info,
                    .pViewportState = &viewport_info,
                    .pRasterizationState = &raster_info,
                    .pMultisampleState = &sample_info,
                    .pDepthStencilState = &depth_info,
                    .pColorBlendState = &blend_info,
                    .layout = pipeline_layout,
                    .renderPass = veekay::app.vk_render_pass,
            };

            vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &pipeline);
        }

        // Создаем буферы
        scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * aligned_sizeof, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        point_lights_buffer = new veekay::graphics::Buffer(
                max_point_lights * sizeof(PointLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // Создаем сэмплер для текстур
        {
            VkSamplerCreateInfo sInfo{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            };
            vkCreateSampler(device, &sInfo, nullptr, &material_sampler);
        }

        // Создаем "заглушку" для текстуры
        {
            VkSamplerCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            };
            vkCreateSampler(device, &info, nullptr, &missing_texture_sampler);

            uint32_t pixels[] = { 0xffff00ff, 0xff00ff00, 0xff0000ff, 0xffffffff };
            missing_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_B8G8R8A8_UNORM, pixels);
        }

        // Обновляем descriptor set для сцены
        {
            VkDescriptorBufferInfo buffer_infos[] = {
                    {scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms)},
                    {model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms)},
                    {point_lights_buffer->buffer, 0, max_point_lights * sizeof(PointLight)},
            };

            VkWriteDescriptorSet write_infos[] = {
                    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buffer_infos[0], nullptr},
                    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &buffer_infos[1], nullptr},
                    {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[2], nullptr},
            };

            vkUpdateDescriptorSets(device, 3, write_infos, 0, nullptr);
        }

        // Загружаем текстуры и создаем материалы
        materials.resize(max_materials);

        // Материал 0 - для пола (нужно добавить!)
        materials[0].albedo = loadTextureFromPNG(cmd, "./assets/cg.png");
        if (!materials[0].albedo) materials[0].albedo = missing_texture;

        // Материал 1 - для красного куба
        materials[1].albedo = loadTextureFromPNG(cmd, "./assets/image_for_cg1.png");
        if (!materials[1].albedo) materials[1].albedo = missing_texture;

        // Материал 2 - для зеленого куба
        materials[2].albedo = loadTextureFromPNG(cmd, "./assets/image_for_cg2.png");
        if (!materials[2].albedo) materials[2].albedo = missing_texture;

        // Материал 3 - для синего куба
        materials[3].albedo = loadTextureFromPNG(cmd, "./assets/image_for_cg3.png");
        if (!materials[3].albedo) materials[3].albedo = missing_texture;

        // Остальные материалы используют заглушку
        for (uint32_t i = 4; i < max_materials; ++i) {
            materials[i].albedo = missing_texture;
        }

        // Обновляем descriptor set'ы материалов
        for (uint32_t i = 0; i < max_materials; ++i) {
            VkDescriptorImageInfo imageInfo{
                .sampler = material_sampler,
                .imageView = materials[i].albedo->view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            };

            VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = material_descriptor_sets[i],
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfo
            };

            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
            materials[i].descriptor_set = material_descriptor_sets[i];
        }

        // Plane mesh initialization
        {
            std::vector<Vertex> vertices = {
                    {{-5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                    {{5.0f, 0.0f, 5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                    {{5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                    {{-5.0f, 0.0f, -5.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {0, 1, 2, 2, 3, 0};

            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            plane_mesh.indices = uint32_t(indices.size());
        }

        // Cube mesh initialization
        {
            std::vector<Vertex> vertices = {
                // Front face (z = -0.5)
                {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
                
                // Back face (z = +0.5)
                {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
                
                // Right face (x = +0.5)
                {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                
                // Left face (x = -0.5)
                {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                
                // Top face (y = +0.5)
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
                
                // Bottom face (y = -0.5)
                {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                0,1,2, 2,3,0,        // front
                4,5,6, 6,7,4,        // back
                8,9,10, 10,11,8,     // right
                12,13,14, 14,15,12,  // left
                16,17,18, 18,19,16,  // top
                20,21,22, 22,23,20,  // bottom
            };

            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            cube_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            cube_mesh.indices = uint32_t(indices.size());
        }

        // Add models to scene with material IDs
        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {0.94f, 0.90f, 0.92f},
            .shininess = 32.0f,
            .material_id = 0  // материал с травой
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{{2.0f, -0.5f, -1.0f}, {0.8f,0.8f,0.8f}, {0.0f, M_PI/6.0f, 0.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {1.0f, 0.8f, 0.8f},
            .shininess = 128.0f,
            .material_id = 1  // материал для красного куба
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{{-2.0f, -0.5f, -0.5f}, {0.9f,0.9f,0.9f}, {0.0f, M_PI/4.0f, 0.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {0.8f, 1.0f, 0.8f},
            .shininess = 96.0f,
            .material_id = 2  // материал для зеленого куба
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{{0.0f, -0.5f, 1.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {0.8f, 0.8f, 1.0f},
            .shininess = 156.0f,
            .material_id = 3  // материал для синего куба
        });

        // Point lights
        point_lights.emplace_back(PointLight{{2.5f, -1.2f, -0.45f}, 4.0f, {1.0f, 0.95f, 0.85f}});
        point_lights.emplace_back(PointLight{{0.0f, -0.5f, -0.2f}, 4.0f, {1.0f, 0.90f, 0.80f}});
        point_lights.emplace_back(PointLight{{-2.2f, -1.1f, -0.25f}, 4.0f, {0.95f, 0.90f, 1.0f}});
    }

    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        if (material_sampler) vkDestroySampler(device, material_sampler, nullptr);
        if (missing_texture_sampler) vkDestroySampler(device, missing_texture_sampler, nullptr);

        delete missing_texture;
        
        // Освобождаем текстуры материалов
        for (auto &mat : materials) {
            if (mat.albedo && mat.albedo != missing_texture) delete mat.albedo;
        }

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;
        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;
        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;
        delete point_lights_buffer;

        if (scene_set_layout) vkDestroyDescriptorSetLayout(device, scene_set_layout, nullptr);
        if (material_set_layout) vkDestroyDescriptorSetLayout(device, material_set_layout, nullptr);
        if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        // UI (можно оставить как есть или упростить)
        ImGui::Begin("Control");

        ImGui::SeparatorText("Ambient light");
        ImGui::SliderFloat("ambient", &ambient_light, 0.0f, 1.0f);

        ImGui::SeparatorText("Directional light");
        ImGui::SliderFloat("intensity", &directional_light.intensity, 0.0f, 2.0f);
        ImGui::ColorEdit3("color", &directional_light.color.x);
        ImGui::SliderFloat3("direction", &directional_light.direction.x, -1.0f, 1.0f);
        if (ImGui::Button("reset")) {
            directional_light.direction = {0.15f, 1.0f, 0.3f};
        }
        directional_light.direction = veekay::vec3::normalized(directional_light.direction);

        ImGui::SeparatorText("Point lights");
        if (ImGui::Button("new light")) {
            if (point_lights.size() < max_point_lights) {
                point_lights.push_back({camera.position, 5.0f, {1.0f, 0.9f, 0.95f}});
            }
        }
        if (ImGui::Button("clear all")) {
            point_lights.clear();
        }
        
        for (size_t i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(i);
            if (ImGui::CollapsingHeader(("light " + std::to_string(i)).c_str())) {
                PointLight& light = point_lights[i];
                ImGui::ColorEdit3("color", &light.color.x);
                ImGui::SliderFloat3("position", &light.position.x, -10.0f, 10.0f);
                ImGui::SliderFloat("radius", &light.radius, 0.1f, 20.0f);
                if (ImGui::Button("erase")) {
                    point_lights.erase(point_lights.begin() + i);
                    ImGui::PopID();
                    break;
                }
            }
            ImGui::PopID();
        }
        ImGui::End();

        // Camera control
        // Camera control
        {
            using namespace veekay::input;
            if (camera_control) {
                auto move_delta = mouse::cursorDelta();
                camera.rotation.y += move_delta.x * mouse_sens * 0.02f;  // Убрали fmod
                camera.rotation.x += -move_delta.y * mouse_sens * 0.02f; // Инвертировали Y для мыши
                camera.rotation.x = std::clamp(camera.rotation.x, -1.5f, 1.5f);
            }

            veekay::vec3 front = camera.forward();  // Теперь правильное направление
            veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, {0.0f, 1.0f, 0.0f}));
            veekay::vec3 up = {0.0f, 1.0f, 0.0f};

            if (keyboard::isKeyPressed(keyboard::Key::c)) {
                camera_control = !camera_control;
                mouse::setCaptured(camera_control);
            }

            float speed = 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += front * speed;   // Вперед
            if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= front * speed;   // Назад
            if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * speed;   // Вправо
            if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * speed;   // Влево
            if (keyboard::isKeyDown(keyboard::Key::e)) camera.position += up * speed;       // Вверх
            if (keyboard::isKeyDown(keyboard::Key::q)) camera.position -= up * speed;       // Вниз
        }

        // Update uniforms
        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        veekay::vec3 sun_color = directional_light.color * directional_light.intensity;

        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio),
            .view_position = camera.position,
            .ambient_light_intensity = {ambient_light, ambient_light, ambient_light},
            .sun_light_direction = directional_light.direction,
            .sun_light_color = sun_color,
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
            const Model& model = models[i];
            const Mesh& mesh = model.mesh;

            if (current_vb != mesh.vertex_buffer->buffer) {
                current_vb = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vb, &offset);
            }

            if (current_ib != mesh.index_buffer->buffer) {
                current_ib = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_ib, 0, VK_INDEX_TYPE_UINT32);
            }

            uint32_t dynamic_offset = uint32_t(i * aligned_sizeof);
            
            // Биндим scene descriptor set (set = 0)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                   0, 1, &scene_descriptor_set, 1, &dynamic_offset);

            // Биндим material descriptor set по ID материала (set = 1)
            int materialId = model.material_id;
            if (materialId < 0 || materialId >= (int)materials.size()) materialId = 0;
            
            VkDescriptorSet matSet = materials[materialId].descriptor_set;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                   1, 1, &matSet, 0, nullptr);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }
}

int main() {
    return veekay::run({initialize, shutdown, update, render});
}