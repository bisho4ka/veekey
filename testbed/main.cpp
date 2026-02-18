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
    constexpr uint32_t max_spot_lights = 16;
    size_t aligned_sizeof;

    float mouse_sens = 0.75f;
    float ambient_light = 0.025f;
    bool camera_control = false;
    bool animate_lights = true;
    bool use_transform_matrix = false;

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
        uint32_t spot_lights_count;
        uint32_t _pad4[2];
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
        float _pad0;

    } PointLight;

    typedef struct SpotLight {

        veekay::vec3 position;

        // at what distance does the color fade
        float radius;

        veekay::vec3 direction;

        float angle_cos;

        veekay::vec3 color;
        float _pad0;

    } SpotLight;

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

        // NOTE: View matrix of camera (inverse of a transform)
        // from world to local coordinates
        veekay::mat4 view() const;

        // NOTE: View and projection composition
        veekay::mat4 view_projection(float aspect_ratio) const;

        // NOTE: Get camera forward direction
        veekay::vec3 forward() const;

        // NOTE: View matrix from transform matrix
        veekay::mat4 view_transform() const;

        // NOTE: Get camera transformation matrix
        veekay::mat4 transform_matrix() const;
    };

    // NOTE: Scene objects
    inline namespace {

        Camera camera
        {
                .position = {-0.3f, -2.0f, -3.6f},
                .rotation = {0.32f, 0.2f, 0.0f}
        };

        std::vector<Model> models;
        std::vector<PointLight> point_lights;
        std::vector<SpotLight> spot_lights;
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
        veekay::graphics::Buffer* spot_lights_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;

        veekay::graphics::Texture* missing_texture;
        VkSampler missing_texture_sampler;

        veekay::graphics::Texture* texture;
        VkSampler texture_sampler;
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
                                const veekay::vec3& world_y)
                                {
        // camera axis z
        veekay::vec3 axis_z = veekay::vec3::normalized(eye_vtr - target);

        // camera axis x
        veekay::vec3 axis_x = veekay::vec3::normalized(veekay::vec3::cross(world_y, axis_z));

        // camera axis y
        veekay::vec3 axis_y = veekay::vec3::cross(axis_z, axis_x);

        /* Look-At
         * [ Rx Ry Rz 0]     [  1  0  0  -Px ]
         * [ Ux Uy Uz 0]  *  [  0  1  0  -Py ]
         * [ Dx Dy Dz 0]     [  0  0  1  -Pz ]
         * [ 0  0  0  1]     [  0  0  0   1  ]
         * */

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

    // calculate camera view matrix
    veekay::mat4 Camera::view() const {
        if (use_transform_matrix) {
            return view_transform();
        }
        // where camera looks now (vector)
        veekay::vec3 forward = Camera::forward();

        // where camera looks now (point)
        veekay::vec3 target = position + forward;

        // axis y
        veekay::vec3 up = {0.0f, 1.0f, 0.0f};

        return look_at_matrix(position, target, up);
    }

    static veekay::mat4 inverted(const veekay::mat4& m) {
        float inv[16];
        float det;
        int i;

        float m0 = m[0][0], m1 = m[0][1], m2 = m[0][2], m3 = m[0][3];
        float m4 = m[1][0], m5 = m[1][1], m6 = m[1][2], m7 = m[1][3];
        float m8 = m[2][0], m9 = m[2][1], m10 = m[2][2], m11 = m[2][3];
        float m12 = m[3][0], m13 = m[3][1], m14 = m[3][2], m15 = m[3][3];

        inv[0] = m5 * m10 * m15 - m5 * m11 * m14 - m9 * m6 * m15 +
                 m9 * m7 * m14 + m13 * m6 * m11 - m13 * m7 * m10;

        inv[4] = -m4 * m10 * m15 + m4 * m11 * m14 + m8 * m6 * m15 -
                 m8 * m7 * m14 - m12 * m6 * m11 + m12 * m7 * m10;

        inv[8] = m4 * m9 * m15 - m4 * m11 * m13 - m8 * m5 * m15 +
                 m8 * m7 * m13 + m12 * m5 * m11 - m12 * m7 * m9;

        inv[12] = -m4 * m9 * m14 + m4 * m10 * m13 + m8 * m5 * m14 -
                  m8 * m6 * m13 - m12 * m5 * m10 + m12 * m6 * m9;

        inv[1] = -m1 * m10 * m15 + m1 * m11 * m14 + m9 * m2 * m15 -
                 m9 * m3 * m14 - m13 * m2 * m11 + m13 * m3 * m10;

        inv[5] = m0 * m10 * m15 - m0 * m11 * m14 - m8 * m2 * m15 +
                 m8 * m3 * m14 + m12 * m2 * m11 - m12 * m3 * m10;

        inv[9] = -m0 * m9 * m15 + m0 * m11 * m13 + m8 * m1 * m15 -
                 m8 * m3 * m13 - m12 * m1 * m11 + m12 * m3 * m9;

        inv[13] = m0 * m9 * m14 - m0 * m10 * m13 - m8 * m1 * m14 +
                  m8 * m2 * m13 + m12 * m1 * m10 - m12 * m2 * m9;

        inv[2] = m1 * m6 * m15 - m1 * m7 * m14 - m5 * m2 * m15 +
                 m5 * m3 * m14 + m13 * m2 * m7 - m13 * m3 * m6;

        inv[6] = -m0 * m6 * m15 + m0 * m7 * m14 + m4 * m2 * m15 -
                 m4 * m3 * m14 - m12 * m2 * m7 + m12 * m3 * m6;

        inv[10] = m0 * m5 * m15 - m0 * m7 * m13 - m4 * m1 * m15 +
                  m4 * m3 * m13 + m12 * m1 * m7 - m12 * m3 * m5;

        inv[14] = -m0 * m5 * m14 + m0 * m6 * m13 + m4 * m1 * m14 -
                  m4 * m2 * m13 - m12 * m1 * m6 + m12 * m2 * m5;

        inv[3] = -m1 * m6 * m11 + m1 * m7 * m10 + m5 * m2 * m11 -
                 m5 * m3 * m10 - m9 * m2 * m7 + m9 * m3 * m6;

        inv[7] = m0 * m6 * m11 - m0 * m7 * m10 - m4 * m2 * m11 +
                 m4 * m3 * m10 + m8 * m2 * m7 - m8 * m3 * m6;

        inv[11] = -m0 * m5 * m11 + m0 * m7 * m9 + m4 * m1 * m11 -
                  m4 * m3 * m9 - m8 * m1 * m7 + m8 * m3 * m5;

        inv[15] = m0 * m5 * m10 - m0 * m6 * m9 - m4 * m1 * m10 +
                  m4 * m2 * m9 + m8 * m1 * m6 - m8 * m2 * m5;

        det = m0 * inv[0] + m1 * inv[4] + m2 * inv[8] + m3 * inv[12];

        if (det == 0) {
            return veekay::mat4::identity();
        }

        det = 1.0f / det;

        veekay::mat4 result;
        for (i = 0; i < 16; i++) {
            result[i / 4][i % 4] = inv[i] * det;
        }

        return result;
    }

    // calculate combined view and projection matrix
    veekay::mat4 Camera::view_projection(float aspect_ratio) const {

        auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

        return view() * projection;
    }

    veekay::mat4 Camera::transform_matrix() const {
        veekay::mat4 result = veekay::mat4::identity();

        result = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y) * result;
        result = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, -rotation.x) * result;
        result = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z) * result;


        result = result * veekay::mat4::translation(position) ;

        return result;
    }

    veekay::mat4 Camera::view_transform() const {
        veekay::mat4 camera_transform = transform_matrix();
        return inverted(camera_transform);
    }

    // NOTE: Loads shader byte code from file
    // NOTE: Your shaders are compiled via CMake with this code too, look it up
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
        if (vkCreateShaderModule(veekay::app.vk_device, &
                info, nullptr, &result) != VK_SUCCESS) {
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

            // NOTE: Vertex shader stage
            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertex_shader_module,
                    .pName = "main",
            };

            // NOTE: Fragment shader stage
            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragment_shader_module,
                    .pName = "main",
            };

            // NOTE: How many bytes does a vertex take?
            VkVertexInputBindingDescription buffer_binding{
                    .binding = 0,
                    .stride = sizeof(Vertex),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            // NOTE: Declare vertex attributes
            VkVertexInputAttributeDescription attributes[] = {
                    {
                            .location = 0, // NOTE: First attribute
                            .binding = 0, // NOTE: First vertex buffer
                            .format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
                            .offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
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

            // NOTE: Describe inputs
            VkPipelineVertexInputStateCreateInfo input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
                    .pVertexAttributeDescriptions = attributes,
            };

            // NOTE: Every three vertices make up a triangle,
            //       so our vertex buffer contains a "list of triangles"
            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            // NOTE: Declare clockwise triangle order as front-facing
            //       Discard triangles that are facing away
            //       Fill triangles, don't draw lines instaed
            VkPipelineRasterizationStateCreateInfo raster_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    .polygonMode = VK_POLYGON_MODE_FILL,
                    .cullMode = VK_CULL_MODE_BACK_BIT,
                    .frontFace = VK_FRONT_FACE_CLOCKWISE,
                    .lineWidth = 1.0f,
            };

            // NOTE: Use 1 sample per pixel
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

            // NOTE: Let rasterizer draw on the entire window
            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

                    .viewportCount = 1,
                    .pViewports = &viewport,

                    .scissorCount = 1,
                    .pScissors = &scissor,
            };

            // NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
            VkPipelineDepthStencilStateCreateInfo depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    .depthTestEnable = true,
                    .depthWriteEnable = true,
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            // NOTE: Let fragment shader write all the color channels
            VkPipelineColorBlendAttachmentState attachment_info{
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT,
            };

            // NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
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

            // NOTE: Descriptor set layout specification
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
                        {
                                .binding = 3,
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

            // NOTE: Declare external data sources, only push constants this time
            VkPipelineLayoutCreateInfo layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .setLayoutCount = 1,
                    .pSetLayouts = &descriptor_set_layout,
            };

            // NOTE: Create pipeline layout
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

            // NOTE: Create graphics pipeline
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

        spot_lights_buffer = new veekay::graphics::Buffer(
                max_spot_lights * sizeof(SpotLight),
                nullptr,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);


        // NOTE: This texture and sampler is used when texture could not be loaded
        {
            VkSamplerCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            };

            if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan texture sampler\n";
                veekay::app.running = false;
                return;
            }

            uint32_t pixels[] = {
                    0xff000000, 0xffff00ff,
                    0xffff00ff, 0xff000000,
            };

            missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
                                                            VK_FORMAT_B8G8R8A8_UNORM,
                                                            pixels);
        }

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
                    {
                            .buffer = spot_lights_buffer->buffer,
                            .offset = 0,
                            .range = max_spot_lights * sizeof(SpotLight),
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
                    {
                            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            .dstSet = descriptor_set,
                            .dstBinding = 3,
                            .dstArrayElement = 0,
                            .descriptorCount = 1,
                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            .pBufferInfo = &buffer_infos[3],
                    },
            };

            vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
                                   write_infos, 0, nullptr);
        }

        // NOTE: Plane mesh initialization
        {
            // (v0)------(v1)
            //  |  \       |
            //  |   `--,   |
            //  |       \  |
            // (v3)------(v2)
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
                .transform = Transform
                        {
                        .position = {2.0f, -0.5f, -1.0f},
                        .scale = {0.8f, 0.8f, 0.8f},
                        .rotation = {0.0f, M_PI/6.0f, 0.0f},
                },
                .albedo_color = {0.95f, 0.88f, 0.75f},
                .specular_color = {0.90f, 0.92f, 0.95f},
                .shininess = 128.0f,
        });

        models.emplace_back(Model {
                .mesh = cube_mesh,
                .transform = Transform
                        {
                        .position = {-2.0f, -0.5f, -0.5f},
                        .scale = {0.9f, 0.9f, 0.9f},
                        .rotation = {0.0f, M_PI/4.0f, 0.0f},
                },
                .albedo_color = {0.88f, 0.95f, 0.85f},
                .specular_color = {0.92f, 0.95f, 0.90f},
                .shininess = 96.0f,
        });

        models.emplace_back(Model {
                .mesh = cube_mesh,
                .transform = Transform
                        {
                        .position = {0.0f, -0.5f, 1.0f},
                },
                .albedo_color = {0.92f, 0.85f, 0.95f},
                .specular_color = {0.98f, 0.94f, 0.90f},
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

        spot_lights.emplace_back(SpotLight {
                .position = camera.position,
                .radius = 6.0f,
                .direction = veekay::vec3::normalized({0.0, 0.5, 0.5}),
                .angle_cos = cos(toRadians(50.0f)),
                .color = veekay::vec3{1.0f, 0.98f, 0.96f},
        });
    };

// NOTE: Destroy resources here, do not cause leaks in your program!
    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        vkDestroySampler(device, missing_texture_sampler, nullptr);
        delete missing_texture;

        delete cube_mesh.index_buffer;
        delete cube_mesh.vertex_buffer;

        delete plane_mesh.index_buffer;
        delete plane_mesh.vertex_buffer;

        delete model_uniforms_buffer;
        delete scene_uniforms_buffer;

        delete point_lights_buffer;
        delete spot_lights_buffer;

        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    void update(double time) {
        ImGui::Begin("CONTROLS");

        ImGui::SeparatorText("CAMERA");
        ImGui::Text("camera position: %.2f, %.2f, %.2f",
                    camera.position.x, camera.position.y, camera.position.z);

        ImGui::SeparatorText("CAMERA MATRIX METHOD");
        ImGui::Checkbox("Use Transform Matrix", &use_transform_matrix);
        if (use_transform_matrix) {
            ImGui::Text("Using: Transform Matrix (Inverse)");
        } else {
            ImGui::Text("Using: Look-At Matrix");
        }

        ImGui::SeparatorText("VIEW");
        ImGui::Checkbox("animate lights ?", &animate_lights);


        ImGui::SeparatorText("AMBIENT LIGHT SETTINGS");
        ImGui::SliderFloat("ambient light", &ambient_light, 0.0f, 1.0f);

        ImGui::SeparatorText("DIRECTIONAL LIGHT SETTINGS");

        ImGui::SliderFloat("intensity", &directional_light.intensity, 0.0f, 2.0f);
        ImGui::ColorEdit3("color", &directional_light.color.x);

        ImGui::SliderFloat3("direction", &directional_light.direction.x, -1.0f, 1.0f);
        if (ImGui::Button("reset direction")) {
            directional_light.direction = {0.15f, 1.0f, 0.3f};
        }
        directional_light.direction = veekay::vec3::normalized(directional_light.direction);

        ImGui::Text("normalized direction: %.2f, %.2f, %.2f",
                    directional_light.direction.x, directional_light.direction.y, directional_light.direction.z);


        ImGui::SeparatorText("POINT LIGHT SETTINGS");
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
        ImGui::SeparatorText("SPOT LIGHT SETTINGS");

        if (ImGui::Button("new spot light")) {
            if (spot_lights.size() < max_spot_lights) {
                SpotLight new_light{
                        .position = camera.position,
                        .radius = 10.0f,
                        .direction = -camera.forward(),
                        .angle_cos = cos(toRadians(30.0f)),
                        .color = {1.0f, 0.9f, 0.95f}
                };
                spot_lights.push_back(new_light);
            }
        }

        if (ImGui::Button("clear spot lights")) {
            spot_lights.clear();
        }

        if (!spot_lights.empty()) {
            ImGui::Separator();
            for (size_t i = 0; i < spot_lights.size(); ++i) {
                ImGui::PushID(static_cast<int>(i + 100));
                if (ImGui::CollapsingHeader(("light " + std::to_string(i)).c_str())) {
                    SpotLight& light = spot_lights[i];
                    ImGui::ColorEdit3("color", &light.color.x);
                    ImGui::SliderFloat3("position", &light.position.x, -10.0f, 10.0f);
                    ImGui::SliderFloat("radius", &light.radius, 0.1f, 20.0f);
                    float angle_degrees = acos(light.angle_cos) * 180.0f / M_PI;
                    if (ImGui::SliderFloat("angle", &angle_degrees, 1.0f, 89.0f)) {
                        light.angle_cos = cos(toRadians(angle_degrees));
                    }
                    if (ImGui::Button("erase")) {
                        spot_lights.erase(spot_lights.begin() + i);
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

            auto view = camera.view();

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


        if (animate_lights) {
            for (size_t i = 0; i < point_lights.size(); ++i) {
                point_lights[i].position.x += sinf(time + i * M_PI_4) / 300.0f;
                point_lights[i].position.z += cosf(time + i * M_PI_4) / 300.0f;
            }
            for (size_t i = 0; i < spot_lights.size(); ++i) {
                spot_lights[i].direction.x += sinf(time + i * M_PI_4) / 400.0f;
                spot_lights[i].direction.z += cosf(time + i * M_PI_4) / 400.0f;
            }
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
                .spot_lights_count = static_cast<uint32_t>(spot_lights.size())
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

        if (point_lights_buffer) {
            if (!point_lights.empty()) {
                std::memcpy(point_lights_buffer->mapped_region,
                            point_lights.data(),
                            point_lights.size() * sizeof(PointLight));
            }
        }

        if (spot_lights_buffer) {
            if (!spot_lights.empty())
            {
                std::memcpy(spot_lights_buffer->mapped_region,
                            spot_lights.data(),
                            spot_lights.size() * sizeof(SpotLight));
            }
        }
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        { // NOTE: Start recording rendering commands
            VkCommandBufferBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(cmd, &info);
        }

        { // NOTE: Use current swapchain framebuffer and clear it
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

        const size_t model_uniorms_alignment =
                veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

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