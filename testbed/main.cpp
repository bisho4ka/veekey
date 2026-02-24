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
        veekay::vec2 uv;
    };

    // scene data structure
    struct SceneUniforms {
        // view & projection matrix
        veekay::mat4 view_projection;

        // camera position
        veekay::vec3 view_position;
        float _pad0;

        // RGB intensity
        veekay::vec3 ambient_light_intensity;
        float _pad1;

        // light parameters
        veekay::vec3 sun_light_direction;
        float _pad2;
        veekay::vec3 sun_light_color;
        float _pad3;

        // count of lights
        uint32_t point_lights_count;
        uint32_t _pad4[3];
    };

    // model data structure
    struct ModelUniforms {
        // matrix from local to world coordinates
        veekay::mat4 model;

        // diffuse color
        veekay::vec3 albedo_color;
        float _pad0;

        // spectacular highlights color
        veekay::vec3 specular_color;
        float _pad2;

        float shininess;
        uint32_t _pad3[3];
    };

    typedef struct PointLight {
        veekay::vec3 position;
        // at what distance does the color fade
        float radius;
        veekay::vec3 color;
        float ambient_intensity;
        float _pad0[3];
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

        // NOTE: Model matrix (translation, rotation and scaling)
        veekay::mat4 matrix() const;
    };

    // all components in one
    struct Model {
        Mesh mesh;
        Transform transform;

        // main color
        veekay::vec3 albedo_color;

        // highlight color
        veekay::vec3 specular_color;
        float shininess;
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

        // NOTE: View matrix of camera using Look-At
        veekay::mat4 view() const;

        // NOTE: View and projection composition
        veekay::mat4 view_projection(float aspect_ratio) const;

        // NOTE: Get camera forward direction
        veekay::vec3 forward() const;
    };

    // NOTE: Scene objects
    inline namespace {
        Camera camera {
            .position = {-0.3f, -2.0f, -3.6f},
            .rotation = {0.32f, 0.2f, 0.0f}
        };

        std::vector<Model> models;
        std::vector<PointLight> point_lights;
    }

    // NOTE: Vulkan objects
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
        // scaling
        veekay::mat4 scale_mat = veekay::mat4::scaling(scale);

        // rotation relative to axis Y
        veekay::mat4 y_rot = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);

        // rotation relative to axis X
        veekay::mat4 x_rot = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);

        // rotation relative to axis Z
        veekay::mat4 z_rot = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);

        // composition of rotations
        veekay::mat4 rotation_mat = y_rot * x_rot * z_rot;

        // translation
        veekay::mat4 translation_mat = veekay::mat4::translation(position);
        return translation_mat * rotation_mat * scale_mat;
    }

    veekay::mat4 look_at_matrix(const veekay::vec3& eye_vtr,
                                const veekay::vec3& target,
                                const veekay::vec3& world_y) {
        // camera axis z - направление взгляда (от камеры к цели)
        // В правой системе координат камера смотрит в направлении -Z
        veekay::vec3 axis_z = veekay::vec3::normalized(eye_vtr - target);  // ВОЗВРАЩАЕМ ОБРАТНО!
        
        // camera axis x
        veekay::vec3 axis_x = veekay::vec3::normalized(veekay::vec3::cross(world_y, axis_z));
        
        // camera axis y
        veekay::vec3 axis_y = veekay::vec3::cross(axis_z, axis_x);
        
        veekay::mat4 result_look_at;
        result_look_at[0][0] = axis_x.x;
        result_look_at[0][1] = axis_y.x;
        result_look_at[0][2] = axis_z.x;
        result_look_at[0][3] = 0.0f;
        
        result_look_at[1][0] = axis_x.y;
        result_look_at[1][1] = axis_y.y;
        result_look_at[1][2] = axis_z.y;
        result_look_at[1][3] = 0.0f;
        
        result_look_at[2][0] = axis_x.z;
        result_look_at[2][1] = axis_y.z;
        result_look_at[2][2] = axis_z.z;
        result_look_at[2][3] = 0.0f;
        
        result_look_at[3][0] = -veekay::vec3::dot(axis_x, eye_vtr);
        result_look_at[3][1] = -veekay::vec3::dot(axis_y, eye_vtr);
        result_look_at[3][2] = -veekay::vec3::dot(axis_z, eye_vtr);
        result_look_at[3][3] = 1.0f;
        
        return result_look_at;
    }

    // calculate the vector of the camera's gaze direction
    veekay::vec3 Camera::forward() const {
        float y_rad = rotation.y;
        float x_rad = rotation.x;

        float cos_y = cosf(y_rad);
        float sin_y = sinf(y_rad);

        // negative trigonometric
        float cos_x = -cosf(x_rad);
        float sin_x = -sinf(x_rad);

        // uses spherical coordinates
        veekay::vec3 result_forward = {sin_y * cos_x, sin_x, cos_y * cos_x};

        result_forward = veekay::vec3::normalized(result_forward);

        return result_forward;
    }

    // calculate camera view matrix using Look-At
    veekay::mat4 Camera::view() const {
        // where camera looks now (vector)
        veekay::vec3 forward = Camera::forward();

        // where camera looks now (point)
        veekay::vec3 target = position + forward;

        // axis y
        veekay::vec3 up = {0.0f, 1.0f, 0.0f};

        return look_at_matrix(position, target, up);
    }

    // calculate combined view and projection matrix
    veekay::mat4 Camera::view_projection(float aspect_ratio) const {
        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);
        return view() * projection;
    }

    // NOTE: Loads shader byte code from file
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
        if (vkCreateShaderModule(veekay::app.vk_device, &info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }

        return result;
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_device, &props);
        uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
        aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;

        { // NOTE: Build graphics pipeline
            vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
            if (!vertex_shader_module) {
                std::cerr << "Failed to load Vulkan vertex shader from file\n";
                veekay::app.running = false;
                return;
            }

            fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
            if (!fragment_shader_module) {
                std::cerr << "Failed to load Vulkan fragment shader from file\n";
                veekay::app.running = false;
                return;
            }

            VkPipelineShaderStageCreateInfo stage_infos[2];

            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertex_shader_module,
                    .pName = "main",
            };

            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragment_shader_module,
                    .pName = "main",
            };

            VkVertexInputBindingDescription buffer_binding{
                    .binding = 0,
                    .stride = sizeof(Vertex),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            VkVertexInputAttributeDescription attributes[] = {
                    {
                            .location = 0,
                            .binding = 0,
                            .format = VK_FORMAT_R32G32B32_SFLOAT,
                            .offset = offsetof(Vertex, position),
                    },
                    {
                            .location = 1,
                            .binding = 0,
                            .format = VK_FORMAT_R32G32B32_SFLOAT,
                            .offset = offsetof(Vertex, normal),
                    },
                    {
                            .location = 2,
                            .binding = 0,
                            .format = VK_FORMAT_R32G32_SFLOAT,
                            .offset = offsetof(Vertex, uv),
                    },
            };

            VkPipelineVertexInputStateCreateInfo input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
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
                    .sampleShadingEnable = false,
                    .minSampleShading = 1.0f,
            };

            VkViewport viewport{
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(veekay::app.window_width),
                    .height = static_cast<float>(veekay::app.window_height),
                    .minDepth = 0.0f,
                    .maxDepth = 1.0f,
            };

            VkRect2D scissor{
                    .offset = {0, 0},
                    .extent = {veekay::app.window_width, veekay::app.window_height},
            };

            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    .viewportCount = 1,
                    .pViewports = &viewport,
                    .scissorCount = 1,
                    .pScissors = &scissor,
            };

            VkPipelineDepthStencilStateCreateInfo depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            VkPipelineColorBlendAttachmentState attachment_info{
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT,
            };

            VkPipelineColorBlendStateCreateInfo blend_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    .logicOpEnable = false,
                    .logicOp = VK_LOGIC_OP_COPY,
                    .attachmentCount = 1,
                    .pAttachments = &attachment_info
            };

            {
                VkDescriptorPoolSize pools[] = {
                        {
                                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                .descriptorCount = 8,
                        },
                        {
                                .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                .descriptorCount = 8,
                        },
                        {
                                .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 8,
                        }
                };

                VkDescriptorPoolCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                        .maxSets = 1,
                        .poolSizeCount = sizeof(pools) / sizeof(pools[0]),
                        .pPoolSizes = pools,
                };

                if (vkCreateDescriptorPool(device, &info, nullptr,
                                           &descriptor_pool) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor pool\n";
                    veekay::app.running = false;
                    return;
                }
            }

            {
                VkDescriptorSetLayoutBinding bindings[] = {
                        {
                                .binding = 0,
                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        },
                        {
                                .binding = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                        },
                        {
                                .binding = 2,
                                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                .descriptorCount = 1,
                                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
                        },
                };

                VkDescriptorSetLayoutCreateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                        .bindingCount = sizeof(bindings) / sizeof(bindings[0]),
                        .pBindings = bindings,
                };

                if (vkCreateDescriptorSetLayout(device, &info, nullptr,
                                                &descriptor_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set layout\n";
                    veekay::app.running = false;
                    return;
                }
            }

            {
                VkDescriptorSetAllocateInfo info{
                        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        .descriptorPool = descriptor_pool,
                        .descriptorSetCount = 1,
                        .pSetLayouts = &descriptor_set_layout,
                };

                if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
                    std::cerr << "Failed to create Vulkan descriptor set\n";
                    veekay::app.running = false;
                    return;
                }
            }

            VkPipelineLayoutCreateInfo layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 1,
                    .pSetLayouts = &descriptor_set_layout,
            };

            if (vkCreatePipelineLayout(device, &layout_info,
                                       nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline layout\n";
                veekay::app.running = false;
                return;
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

            if (vkCreateGraphicsPipelines(device, nullptr,
                                          1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
                nullptr,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        point_lights_buffer = new veekay::graphics::Buffer(
                max_point_lights * sizeof(PointLight),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        {
            VkDescriptorBufferInfo buffer_infos[] = {
                    {
                            .buffer = scene_uniforms_buffer->buffer,
                            .offset = 0,
                            .range = sizeof(SceneUniforms),
                    },
                    {
                            .buffer = model_uniforms_buffer->buffer,
                            .offset = 0,
                            .range = sizeof(ModelUniforms),
                    },
                    {
                            .buffer = point_lights_buffer->buffer,
                            .offset = 0,
                            .range = max_point_lights * sizeof(PointLight),
                    },
            };

            VkWriteDescriptorSet write_infos[] = {
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 0,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            .pBufferInfo = &buffer_infos[0],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 1,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                            .pBufferInfo = &buffer_infos[1],
                    },
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 2,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .pBufferInfo = &buffer_infos[2],
                    },
            };

            vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                                   write_infos, 0, nullptr);
        }

        // NOTE: Plane mesh initialization
        {
            std::vector<Vertex> vertices = {
                    {{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                    {{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                    {{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                    {{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };

            std::vector<uint32_t> indices = {
                    0, 1, 2, 2, 3, 0
            };

            plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            plane_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            plane_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Cube mesh initialization
        {
            std::vector<Vertex> vertices = {
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

            std::vector<uint32_t> indices = {
                    0, 1, 2, 2, 3, 0,
                    4, 5, 6, 6, 7, 4,
                    8, 9, 10, 10, 11, 8,
                    12, 13, 14, 14, 15, 12,
                    16, 17, 18, 18, 19, 16,
                    20, 21, 22, 22, 23, 20,
            };

            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(),
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

            cube_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(),
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

            cube_mesh.indices = uint32_t(indices.size());
        }

        // NOTE: Add models to scene
        models.emplace_back(Model {
            .mesh = plane_mesh,
            .transform = Transform{},
            .albedo_color = {0.98f, 0.85f, 0.90f},
            .specular_color = {0.94f, 0.90f, 0.92f},
            .shininess = 32.0f,
    });

        models.emplace_back(Model {
            .mesh = cube_mesh,
            .transform = Transform {
                    .position = {2.0f, -0.5f, -1.0f},
                    .scale = {0.8f, 0.8f, 0.8f},
                    .rotation = {0.0f, M_PI/6.0f, 0.0f},
            },
            .albedo_color = {1.0f, 0.2f, 0.2f},    
            .specular_color = {1.0f, 0.8f, 0.8f}, 
            .shininess = 128.0f,
        });


        models.emplace_back(Model {
                .mesh = cube_mesh,
                .transform = Transform {
                        .position = {-2.0f, -0.5f, -0.5f},
                        .scale = {0.9f, 0.9f, 0.9f},
                        .rotation = {0.0f, M_PI/4.0f, 0.0f},
                },
                .albedo_color = {0.2f, 1.0f, 0.2f}, 
                .specular_color = {0.8f, 1.0f, 0.8f}, 
                .shininess = 96.0f,
        });

        models.emplace_back(Model {
                .mesh = cube_mesh,
                .transform = Transform {
                        .position = {0.0f, -0.5f, 1.0f},
                },
                .albedo_color = {0.2f, 0.2f, 1.0f},  
                .specular_color = {0.8f, 0.8f, 1.0f},   
                .shininess = 156.0f,
        });


        point_lights.emplace_back(PointLight {
                .position = veekay::vec3{2.5f, -1.2f, -0.45f},
                .radius = 4.0f,
                .color = veekay::vec3{1.0f, 0.95f, 0.85f}
        });

        point_lights.emplace_back(PointLight {
                .position = veekay::vec3{0.0f, -0.5f, -0.2f},
                .radius = 4.0f,
                .color = veekay::vec3{1.0f, 0.90f, 0.80f},
        });

        point_lights.emplace_back(PointLight {
                .position = veekay::vec3{-2.2f, -1.1f, -0.25f},
                .radius = 4.0f,
                .color = veekay::vec3{0.95f, 0.90f, 1.0f},
        });
    }

// NOTE: Destroy resources here, do not cause leaks in your program!
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
        ImGui::Begin("Control");

        ImGui::SeparatorText("Ambient light settings");
        ImGui::SliderFloat("ambient light", &ambient_light, 0.0f, 1.0f);

        ImGui::SeparatorText("Directional light settings");

        ImGui::SliderFloat("intensity", &directional_light.intensity, 0.0f, 2.0f);
        ImGui::ColorEdit3("color", &directional_light.color.x);

        ImGui::SliderFloat3("direction", &directional_light.direction.x, -1.0f, 1.0f);
        if (ImGui::Button("reset direction")) {
            directional_light.direction = {0.15f, 1.0f, 0.3f};
        }
        directional_light.direction = veekay::vec3::normalized(directional_light.direction);

        ImGui::SeparatorText("Point light settings");
        if (ImGui::Button("new point light")) {
            if (point_lights.size() < max_point_lights) {
                PointLight new_light{
                        .position = camera.position,
                        .radius = 5.0f,
                        .color = {1.0f, 0.9f, 0.95f}
                };
                point_lights.push_back(new_light);
            }
        }

        if (ImGui::Button("clear point lights")) {
            point_lights.clear();
        }
        if (!point_lights.empty()) {
            ImGui::Separator();

            for (size_t i = 0; i < point_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i));
                if (ImGui::CollapsingHeader(("light " + std::to_string(i)).c_str())) {
                    PointLight& light = point_lights[i];
                    ImGui::ColorEdit3("color", &light.color.x);
                    ImGui::SliderFloat("ambient intensity", &light.ambient_intensity, 0.0f, 1.0f);
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
        }
        ImGui::End();

        {
            using namespace veekay::input;
            if (camera_control) {
                auto move_delta = mouse::cursorDelta();

                camera.rotation.y = fmod(2.0f * M_PI + camera.rotation.y + move_delta.x * mouse_sens * 0.02f, 2.0f * M_PI);
                camera.rotation.x = fmod(camera.rotation.x + move_delta.y * mouse_sens * 0.02f, 2.0f * M_PI);
                camera.rotation.x = std::clamp(camera.rotation.x, -0.96f * float(M_PI_2), 0.96f * float(M_PI_2));
            }

            veekay::vec3 right = {0.1f * cos(camera.rotation.y), 0.0f, -0.1f * sin(camera.rotation.y)};
            veekay::vec3 up = {0.0f, -0.1f, 0.0f};
            veekay::vec3 front = {0.1f * sin(camera.rotation.y), 0.0f, 0.1f * cos(camera.rotation.y)};

            if (keyboard::isKeyPressed(keyboard::Key::c)) {
                camera_control = !camera_control;
                mouse::setCaptured(camera_control);
            }

            if (keyboard::isKeyDown(keyboard::Key::w))
                camera.position += front;

            if (keyboard::isKeyDown(keyboard::Key::s))
                camera.position -= front;

            if (keyboard::isKeyDown(keyboard::Key::d))
                camera.position += right;

            if (keyboard::isKeyDown(keyboard::Key::a))
                camera.position -= right;

            if (keyboard::isKeyDown(keyboard::Key::q))
                camera.position += up;

            if (keyboard::isKeyDown(keyboard::Key::z))
                camera.position -= up;
        }

        veekay::vec3 sun_direction, sun_color;
        float ambient_level = ambient_light;
        sun_direction = directional_light.direction;
        sun_color = directional_light.color * directional_light.intensity;

        ambient_level *= 0.4;

        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);

        SceneUniforms scene_uniforms{
                .view_projection = camera.view_projection(aspect_ratio),
                .view_position = camera.position,
                .ambient_light_intensity = {ambient_level, ambient_level, ambient_level},
                .sun_light_direction = sun_direction,
                .sun_light_color = sun_color,
                .point_lights_count = static_cast<uint32_t>(point_lights.size()),
        };

        std::vector<ModelUniforms> model_uniforms(models.size());

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model& model = models[i];
            ModelUniforms& uniforms = model_uniforms[i];

            uniforms.model = model.transform.matrix();
            uniforms.albedo_color = model.albedo_color;
            uniforms.specular_color = model.specular_color;
            uniforms.shininess = model.shininess;
        }

        *(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

        uint8_t* base = static_cast<uint8_t*>(model_uniforms_buffer->mapped_region);

        for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
            std::memcpy(base + i * aligned_sizeof, &model_uniforms[i], sizeof(ModelUniforms));
        }

        if (point_lights_buffer && !point_lights.empty()) {
            std::memcpy(point_lights_buffer->mapped_region,
                        point_lights.data(),
                        point_lights.size() * sizeof(PointLight));
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        {
            VkCommandBufferBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(cmd, &info);
        }

        {
            VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

            VkClearValue clear_values[] = {clear_color, clear_depth};

            VkRenderPassBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    .renderPass = veekay::app.vk_render_pass,
                    .framebuffer = framebuffer,
                    .renderArea = {
                            .extent = {
                                    veekay::app.window_width,
                                    veekay::app.window_height
                            },
                    },
                    .clearValueCount = 2,
                    .pClearValues = clear_values,
            };

            vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
        }

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize zero_offset = 0;

        VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
        VkBuffer current_index_buffer = VK_NULL_HANDLE;

        for (size_t i = 0, n = models.size(); i < n; ++i) {
            const Model& model = models[i];
            const Mesh& mesh = model.mesh;

            if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
                current_vertex_buffer = mesh.vertex_buffer->buffer;
                vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
            }

            if (current_index_buffer != mesh.index_buffer->buffer) {
                current_index_buffer = mesh.index_buffer->buffer;
                vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
            }

            uint32_t offset = uint32_t(i * aligned_sizeof);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
                                    0, 1, &descriptor_set, 1, &offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace

int main() {
    std::srand(time(nullptr));
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}