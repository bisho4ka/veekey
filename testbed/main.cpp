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
#include <lodepng.h>

#ifndef M_PI
constexpr float M_PI = 3.14159265358979323846f;
#endif

namespace {
    constexpr uint32_t max_models = 1024;
    constexpr uint32_t max_point_lights = 16;
    constexpr uint32_t max_materials = 8;
    constexpr uint32_t shadow_map_size = 2048;  // размер карт теней
    
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

    // scene data structure (добавлены матрицы для теней)
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
        veekay::mat4 dir_light_matrix;      // матрица для направленного света
        veekay::mat4 point_light_matrix;     // матрица для точечного света
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
        int material_id = 0;
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
        VkShaderModule shadow_vertex_shader_module;  // для рендера теней

        VkDescriptorPool descriptor_pool;
        
        VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout material_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout shadow_set_layout = VK_NULL_HANDLE;  // для теневого прохода
        
        VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> material_descriptor_sets;
        VkDescriptorSet shadow_descriptor_set = VK_NULL_HANDLE;  // для теневого прохода

        VkPipelineLayout pipeline_layout;
        VkPipelineLayout shadow_pipeline_layout;  // layout для теневого пайплайна
        VkPipeline pipeline;
        VkPipeline shadow_pipeline;  // пайплайн для рендера теней

        VkRenderPass shadow_render_pass;  // render pass для теней
        VkSampler shadow_sampler;  // сэмплер с сравнением для теней

        // Ресурсы для карт теней
        VkFramebuffer shadow_dir_framebuffer = VK_NULL_HANDLE;
        VkImage shadow_dir_image = VK_NULL_HANDLE;
        VkDeviceMemory shadow_dir_memory = VK_NULL_HANDLE;
        VkImageView shadow_dir_view = VK_NULL_HANDLE;

        VkFramebuffer shadow_point_framebuffer = VK_NULL_HANDLE;
        VkImage shadow_point_image = VK_NULL_HANDLE;
        VkDeviceMemory shadow_point_memory = VK_NULL_HANDLE;
        VkImageView shadow_point_view = VK_NULL_HANDLE;

        veekay::graphics::Buffer* scene_uniforms_buffer;
        veekay::graphics::Buffer* model_uniforms_buffer;
        veekay::graphics::Buffer* point_lights_buffer;

        Mesh plane_mesh;
        Mesh cube_mesh;

        veekay::graphics::Texture* missing_texture;
        VkSampler missing_texture_sampler;
        VkSampler material_sampler = VK_NULL_HANDLE;

        struct Material {
            veekay::graphics::Texture* albedo = nullptr;
            VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        };
        std::vector<Material> materials;
    }

    // Вспомогательные функции
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

    // Ортографическая матрица для направленного света
    veekay::mat4 mat4_ortho(float left, float right, float bottom, float top, float zNear, float zFar) {
        veekay::mat4 res = veekay::mat4::identity();

        res.columns[0].x = 2.0f / (right - left);
        res.columns[1].y = -2.0f / (bottom - top);
        res.columns[2].z = 1.0f / (zFar - zNear);

        res.columns[3].x = -(right + left) / (right - left);
        res.columns[3].y = (top + bottom) / (bottom - top);
        res.columns[3].z = -zNear / (zFar - zNear);

        return res;
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

    // Поиск подходящего типа памяти
    uint32_t findMemoryType(uint32_t typeFilter) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(veekay::app.vk_physical_device, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                return i;
        }
        return 0;
    }

    // Создание ресурсов для карты глубины
    void createShadowMapResource(VkDevice device, VkImage &image, VkDeviceMemory &mem, VkImageView &view) {
        VkImageCreateInfo imageInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                .imageType = VK_IMAGE_TYPE_2D,
                .format = VK_FORMAT_D32_SFLOAT,
                .extent = {shadow_map_size, shadow_map_size, 1},
                .mipLevels = 1,
                .arrayLayers = 1,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .tiling = VK_IMAGE_TILING_OPTIMAL,
                .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };

        vkCreateImage(device, &imageInfo, nullptr, &image);

        VkMemoryRequirements memReq;
        vkGetImageMemoryRequirements(device, image, &memReq);
        VkMemoryAllocateInfo allocInfo{
                .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                .allocationSize = memReq.size,
                .memoryTypeIndex = findMemoryType(memReq.memoryTypeBits)
        };
        vkAllocateMemory(device, &allocInfo, nullptr, &mem);
        vkBindImageMemory(device, image, mem, 0);

        VkImageViewCreateInfo viewInfo{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = image,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_D32_SFLOAT,
                .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}
        };
        vkCreateImageView(device, &viewInfo, nullptr, &view);
    }

    // Переход состояния изображения
    void transitionShadowImage(VkCommandBuffer cmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkImageMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = (VkAccessFlags)(oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? 
                                VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
            .dstAccessMask = (VkAccessFlags)(newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ? 
                                VK_ACCESS_SHADER_READ_BIT : VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .image = image,
            .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}
        };

        VkPipelineStageFlags srcStage = (oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ? 
                                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        VkPipelineStageFlags dstStage = (newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ? 
                                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    }

    void initialize(VkCommandBuffer cmd) {
        VkDevice& device = veekay::app.vk_device;
        VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physical_device, &props);
        uint32_t alignment = props.limits.minUniformBufferOffsetAlignment;
        aligned_sizeof = ((sizeof(ModelUniforms) + alignment - 1) / alignment) * alignment;

        // Загрузка шейдеров
        vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
        fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
        shadow_vertex_shader_module = loadShaderModule("./shaders/shadow.vert.spv");

        // Создание ресурсов для теней
        {
            // Сэмплер с поддержкой сравнения
            VkSamplerCreateInfo samplerInfo{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,
                .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
                .compareEnable = VK_TRUE,
                .compareOp = VK_COMPARE_OP_LESS,
                .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
            };
            vkCreateSampler(device, &samplerInfo, nullptr, &shadow_sampler);

            // Render pass для теней
            VkAttachmentDescription depthAttachment{
                .format = VK_FORMAT_D32_SFLOAT,
                .samples = VK_SAMPLE_COUNT_1_BIT,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
            };

            VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription subpass{
                .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                .pDepthStencilAttachment = &depthRef
            };

            VkRenderPassCreateInfo rpInfo{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                .attachmentCount = 1,
                .pAttachments = &depthAttachment,
                .subpassCount = 1,
                .pSubpasses = &subpass
            };
            vkCreateRenderPass(device, &rpInfo, nullptr, &shadow_render_pass);

            // Создаем карты глубины
            createShadowMapResource(device, shadow_dir_image, shadow_dir_memory, shadow_dir_view);
            createShadowMapResource(device, shadow_point_image, shadow_point_memory, shadow_point_view);

            // Добавляем начальный переход состояния для карт теней
            VkImageMemoryBarrier barriers[] = {
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .image = shadow_dir_image,
                    .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}
                },
                {
                    .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    .srcAccessMask = 0,
                    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    .image = shadow_point_image,
                    .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}
                }
            };

            vkCmdPipelineBarrier(cmd,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                0, 0, nullptr, 0, nullptr, 2, barriers);

            // Framebuffer для направленного света
            VkFramebufferCreateInfo fbDirInfo{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = shadow_render_pass,
                .attachmentCount = 1,
                .pAttachments = &shadow_dir_view,
                .width = shadow_map_size,
                .height = shadow_map_size,
                .layers = 1
            };
            vkCreateFramebuffer(device, &fbDirInfo, nullptr, &shadow_dir_framebuffer);

            // Framebuffer для точечного света
            VkFramebufferCreateInfo fbPointInfo{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = shadow_render_pass,
                .attachmentCount = 1,
                .pAttachments = &shadow_point_view,
                .width = shadow_map_size,
                .height = shadow_map_size,
                .layers = 1
            };
            vkCreateFramebuffer(device, &fbPointInfo, nullptr, &shadow_point_framebuffer);
        }

        // Создание пула дескрипторов (увеличиваем для теней)
        {
            VkDescriptorPoolSize pools[] = {
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 16},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, max_materials + 2},  // +2 для карт теней
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8},
            };

            VkDescriptorPoolCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                .maxSets = 2 + max_materials,  // scene + shadow + материалы
                .poolSizeCount = 4,
                .pPoolSizes = pools,
            };
            vkCreateDescriptorPool(device, &info, nullptr, &descriptor_pool);
        }

        // Layout для сцены (добавляем binding для карт теней)
        {
            VkDescriptorSetLayoutBinding bindings[] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, nullptr},
                {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_ALL, nullptr},
                {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
                {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},  // dir shadow map
                {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},  // point shadow map
            };

            VkDescriptorSetLayoutCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 5,
                .pBindings = bindings,
            };
            vkCreateDescriptorSetLayout(device, &info, nullptr, &scene_set_layout);
        }

        // Layout для материалов
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

        // Layout для теневого прохода
        {
            VkDescriptorSetLayoutBinding bindings[] = {
                {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
                {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr},
            };

            VkDescriptorSetLayoutCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                .bindingCount = 2,
                .pBindings = bindings,
            };
            vkCreateDescriptorSetLayout(device, &info, nullptr, &shadow_set_layout);
        }

        // Выделяем descriptor set'ы
        {
            VkDescriptorSetAllocateInfo info{
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                .descriptorPool = descriptor_pool,
                .descriptorSetCount = 1,
                .pSetLayouts = &scene_set_layout,
            };
            vkAllocateDescriptorSets(device, &info, &scene_descriptor_set);

            // Для материалов
            material_descriptor_sets.resize(max_materials);
            std::vector<VkDescriptorSetLayout> material_layouts(max_materials, material_set_layout);
            info.pSetLayouts = material_layouts.data();
            info.descriptorSetCount = (uint32_t)material_layouts.size();
            vkAllocateDescriptorSets(device, &info, material_descriptor_sets.data());

            // Для теневого прохода
            info.pSetLayouts = &shadow_set_layout;
            info.descriptorSetCount = 1;
            vkAllocateDescriptorSets(device, &info, &shadow_descriptor_set);
        }

        // Layout основного пайплайна
        {
            VkDescriptorSetLayout setLayouts[] = { scene_set_layout, material_set_layout };
            VkPipelineLayoutCreateInfo layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 2,
                .pSetLayouts = setLayouts,
            };
            vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout);
        }

        // Layout теневого пайплайна с push constant
        {
            VkPushConstantRange pc_range{
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
                .offset = 0,
                .size = sizeof(veekay::mat4)
            };

            VkPipelineLayoutCreateInfo layout_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                .setLayoutCount = 1,
                .pSetLayouts = &shadow_set_layout,
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &pc_range,
            };
            vkCreatePipelineLayout(device, &layout_info, nullptr, &shadow_pipeline_layout);
        }

        // Создание буферов
        scene_uniforms_buffer = new veekay::graphics::Buffer(
                sizeof(SceneUniforms), nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        model_uniforms_buffer = new veekay::graphics::Buffer(
                max_models * aligned_sizeof, nullptr, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

        point_lights_buffer = new veekay::graphics::Buffer(
                max_point_lights * sizeof(PointLight), nullptr, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

        // Создание основного пайплайна
        {
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
                {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)},
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

        // Создание теневого пайплайна
        {
            VkPipelineShaderStageCreateInfo shadow_stage{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT,
                .module = shadow_vertex_shader_module,
                .pName = "main",
            };

            VkVertexInputBindingDescription buffer_binding{
                .binding = 0,
                .stride = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            VkVertexInputAttributeDescription shadow_attributes[] = {
                {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position)},
            };

            VkPipelineVertexInputStateCreateInfo shadow_input_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                .vertexBindingDescriptionCount = 1,
                .pVertexBindingDescriptions = &buffer_binding,
                .vertexAttributeDescriptionCount = 1,
                .pVertexAttributeDescriptions = shadow_attributes,
            };

            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            VkPipelineRasterizationStateCreateInfo shadow_raster_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                .polygonMode = VK_POLYGON_MODE_FILL,
                .cullMode = VK_CULL_MODE_BACK_BIT,
                .frontFace = VK_FRONT_FACE_CLOCKWISE,
                .depthBiasEnable = VK_TRUE,
                .depthBiasConstantFactor = 2.0f,
                .depthBiasSlopeFactor = 2.0f,
                .lineWidth = 1.0f,
            };

            VkPipelineMultisampleStateCreateInfo sample_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            };

            VkViewport shadow_viewport{
                .x = 0.0f, .y = 0.0f,
                .width = static_cast<float>(shadow_map_size),
                .height = static_cast<float>(shadow_map_size),
                .minDepth = 0.0f, .maxDepth = 1.0f,
            };

            VkRect2D shadow_scissor{
                .offset = {0, 0},
                .extent = {shadow_map_size, shadow_map_size},
            };

            VkPipelineViewportStateCreateInfo shadow_viewport_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                .viewportCount = 1, .pViewports = &shadow_viewport,
                .scissorCount = 1, .pScissors = &shadow_scissor,
            };

            VkPipelineDepthStencilStateCreateInfo shadow_depth_info{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                .depthTestEnable = true, .depthWriteEnable = true,
                .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            VkGraphicsPipelineCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                .stageCount = 1,
                .pStages = &shadow_stage,
                .pVertexInputState = &shadow_input_state_info,
                .pInputAssemblyState = &assembly_state_info,
                .pViewportState = &shadow_viewport_info,
                .pRasterizationState = &shadow_raster_info,
                .pMultisampleState = &sample_info,
                .pDepthStencilState = &shadow_depth_info,
                .layout = shadow_pipeline_layout,
                .renderPass = shadow_render_pass,
            };
            vkCreateGraphicsPipelines(device, nullptr, 1, &info, nullptr, &shadow_pipeline);
        }

        // Создание сэмплера для материалов
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

        // Создание текстуры-заглушки
        {
            VkSamplerCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            };
            vkCreateSampler(device, &info, nullptr, &missing_texture_sampler);

            uint32_t pixels[] = { 0xffff00ff, 0xff00ff00, 0xff0000ff, 0xffffffff };
            missing_texture = new veekay::graphics::Texture(cmd, 2, 2, VK_FORMAT_B8G8R8A8_UNORM, pixels);
        }

        // Обновление scene descriptor set с картами теней
        {
            VkDescriptorBufferInfo buffer_infos[] = {
                {scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms)},
                {model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms)},
                {point_lights_buffer->buffer, 0, max_point_lights * sizeof(PointLight)},
            };

            VkDescriptorImageInfo dir_shadow_info{
                .sampler = shadow_sampler,
                .imageView = shadow_dir_view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkDescriptorImageInfo point_shadow_info{
                .sampler = shadow_sampler,
                .imageView = shadow_point_view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            VkWriteDescriptorSet write_infos[] = {
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buffer_infos[0], nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &buffer_infos[1], nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &buffer_infos[2], nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 3, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &dir_shadow_info, nullptr, nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_descriptor_set, 4, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &point_shadow_info, nullptr, nullptr},
            };
            vkUpdateDescriptorSets(device, 5, write_infos, 0, nullptr);
        }

        // Обновление shadow descriptor set
        {
            VkDescriptorBufferInfo buffer_infos[] = {
                {scene_uniforms_buffer->buffer, 0, sizeof(SceneUniforms)},
                {model_uniforms_buffer->buffer, 0, sizeof(ModelUniforms)},
            };

            VkWriteDescriptorSet write_infos[] = {
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, shadow_descriptor_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &buffer_infos[0], nullptr},
                {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, shadow_descriptor_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, nullptr, &buffer_infos[1], nullptr},
            };
            vkUpdateDescriptorSets(device, 2, write_infos, 0, nullptr);
        }

        // Загрузка текстур
        materials.resize(max_materials);
        materials[0].albedo = loadTextureFromPNG(cmd, "./assets/cg.png");
        if (!materials[0].albedo) materials[0].albedo = missing_texture;
        materials[1].albedo = loadTextureFromPNG(cmd, "./assets/image_for_cg1.png");
        if (!materials[1].albedo) materials[1].albedo = missing_texture;
        materials[2].albedo = loadTextureFromPNG(cmd, "./assets/image_for_cg2.png");
        if (!materials[2].albedo) materials[2].albedo = missing_texture;
        materials[3].albedo = loadTextureFromPNG(cmd, "./assets/image_for_cg3.png");
        if (!materials[3].albedo) materials[3].albedo = missing_texture;
        for (uint32_t i = 4; i < max_materials; ++i) {
            materials[i].albedo = missing_texture;
        }

        // Обновление material descriptor set'ов
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

        // Создание мешей
        {
            // Plane mesh
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

            // Cube mesh (ваш существующий код)
            vertices = {
                {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},
                {{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
                {{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                {{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
                {{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
                {{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},
                {{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
                {{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
                {{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
                {{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
                {{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            };
            indices = {
                0,1,2, 2,3,0, 4,5,6, 6,7,4, 8,9,10, 10,11,8,
                12,13,14, 14,15,12, 16,17,18, 18,19,16, 20,21,22, 22,23,20,
            };
            cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
                    vertices.size() * sizeof(Vertex), vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            cube_mesh.index_buffer = new veekay::graphics::Buffer(
                    indices.size() * sizeof(uint32_t), indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
            cube_mesh.indices = uint32_t(indices.size());
        }

        // Добавление моделей (ваши исходные модели)
        models.emplace_back(Model{
            .mesh = plane_mesh,
            .transform = Transform{},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {0.94f, 0.90f, 0.92f},
            .shininess = 32.0f,
            .material_id = 0
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{{2.0f, -0.5f, -1.0f}, {0.8f,0.8f,0.8f}, {0.0f, M_PI/6.0f, 0.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {1.0f, 0.8f, 0.8f},
            .shininess = 128.0f,
            .material_id = 1
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{{-2.0f, -0.5f, -0.5f}, {0.9f,0.9f,0.9f}, {0.0f, M_PI/4.0f, 0.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {0.8f, 1.0f, 0.8f},
            .shininess = 96.0f,
            .material_id = 2
        });

        models.emplace_back(Model{
            .mesh = cube_mesh,
            .transform = Transform{{0.0f, -0.5f, 1.0f}},
            .albedo_color = {1.0f, 1.0f, 1.0f},
            .specular_color = {0.8f, 0.8f, 1.0f},
            .shininess = 156.0f,
            .material_id = 3
        });

        // Точечные источники света
        point_lights.emplace_back(PointLight{{2.5f, -1.2f, -0.45f}, 4.0f, {1.0f, 0.95f, 0.85f}});
        point_lights.emplace_back(PointLight{{0.0f, -0.5f, -0.2f}, 4.0f, {1.0f, 0.90f, 0.80f}});
        point_lights.emplace_back(PointLight{{-2.2f, -1.1f, -0.25f}, 4.0f, {0.95f, 0.90f, 1.0f}});
    }

    void shutdown() {
        VkDevice& device = veekay::app.vk_device;

        // Очистка ресурсов теней
        auto cleanupShadow = [&](VkFramebuffer fb, VkImageView view, VkImage img, VkDeviceMemory mem) {
            if (fb) vkDestroyFramebuffer(device, fb, nullptr);
            if (view) vkDestroyImageView(device, view, nullptr);
            if (img) vkDestroyImage(device, img, nullptr);
            if (mem) vkFreeMemory(device, mem, nullptr);
        };
        cleanupShadow(shadow_dir_framebuffer, shadow_dir_view, shadow_dir_image, shadow_dir_memory);
        cleanupShadow(shadow_point_framebuffer, shadow_point_view, shadow_point_image, shadow_point_memory);

        if (shadow_render_pass) vkDestroyRenderPass(device, shadow_render_pass, nullptr);
        if (shadow_sampler) vkDestroySampler(device, shadow_sampler, nullptr);
        if (shadow_pipeline) vkDestroyPipeline(device, shadow_pipeline, nullptr);
        if (shadow_pipeline_layout) vkDestroyPipelineLayout(device, shadow_pipeline_layout, nullptr);
        if (shadow_set_layout) vkDestroyDescriptorSetLayout(device, shadow_set_layout, nullptr);
        if (shadow_vertex_shader_module) vkDestroyShaderModule(device, shadow_vertex_shader_module, nullptr);

        if (material_sampler) vkDestroySampler(device, material_sampler, nullptr);
        if (missing_texture_sampler) vkDestroySampler(device, missing_texture_sampler, nullptr);

        delete missing_texture;
        
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
        // UI (без изменений)
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
        {
            using namespace veekay::input;
            if (camera_control) {
                auto move_delta = mouse::cursorDelta();
                camera.rotation.y += move_delta.x * mouse_sens * 0.02f;
                camera.rotation.x += move_delta.y * mouse_sens * 0.02f;
                camera.rotation.x = std::clamp(camera.rotation.x, -1.5f, 1.5f);
            }

            veekay::vec3 front = camera.forward();
            veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, {0.0f, 1.0f, 0.0f}));
            veekay::vec3 up = {0.0f, 1.0f, 0.0f};

            if (keyboard::isKeyPressed(keyboard::Key::c)) {
                camera_control = !camera_control;
                mouse::setCaptured(camera_control);
            }

            float speed = 0.1f;
            if (keyboard::isKeyDown(keyboard::Key::s)) camera.position += front * speed;
            if (keyboard::isKeyDown(keyboard::Key::w)) camera.position -= front * speed;
            if (keyboard::isKeyDown(keyboard::Key::d)) camera.position += right * speed;
            if (keyboard::isKeyDown(keyboard::Key::a)) camera.position -= right * speed;
            if (keyboard::isKeyDown(keyboard::Key::e)) camera.position += up * speed;
            if (keyboard::isKeyDown(keyboard::Key::q)) camera.position -= up * speed;
        }

        // Расчет матриц для теней
        float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
        veekay::vec3 sun_color = directional_light.color * directional_light.intensity;

        // Матрица для направленного света (ортографическая проекция)
        veekay::vec3 lightPos = point_lights.empty() ? veekay::vec3{0.0f, 10.0f, 0.0f} : point_lights[0].position;
        veekay::vec3 lightTarget = {0.0f, 0.0f, 0.0f};
        veekay::vec3 lightUp = {0.0f, 0.0f, 1.0f};
        
        veekay::mat4 dirView = look_at_matrix(lightPos, lightTarget, lightUp);
        veekay::mat4 dirProj = mat4_ortho(-10.0f, 10.0f, -10.0f, 10.0f, 1.0f, 20.0f);
        veekay::mat4 dirMatrix = dirView * dirProj;

        // Матрица для точечного света (перспективная проекция)
        veekay::vec3 pointPos = point_lights.empty() ? veekay::vec3{0.0f, 5.0f, 0.0f} : point_lights[0].position;
        veekay::mat4 pointView = look_at_matrix(pointPos, pointPos - directional_light.direction, lightUp);
        veekay::mat4 pointProj = veekay::mat4::projection(90.0f, 1.0f, 0.1f, 20.0f);
        veekay::mat4 pointMatrix = pointView * pointProj;

        SceneUniforms scene_uniforms{
            .view_projection = camera.view_projection(aspect_ratio),
            .view_position = camera.position,
            .ambient_light_intensity = {ambient_light, ambient_light, ambient_light},
            .sun_light_direction = directional_light.direction,
            .sun_light_color = sun_color,
            .dir_light_matrix = dirMatrix,
            .point_light_matrix = pointMatrix,
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

    // Функция для рендера карты теней
    void renderShadowPass(VkCommandBuffer cmd, VkFramebuffer fb, const veekay::mat4& lightMatrix) {
        VkClearValue clear_val{.depthStencil = {1.0f, 0}};
        VkRenderPassBeginInfo rp_info{
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = shadow_render_pass,
            .framebuffer = fb,
            .renderArea = {{0, 0}, {shadow_map_size, shadow_map_size}},
            .clearValueCount = 1,
            .pClearValues = &clear_val
        };

        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline);

        // Push constant с матрицей lightViewProj
        vkCmdPushConstants(cmd, shadow_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(veekay::mat4), &lightMatrix);

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
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_layout,
                                    0, 1, &shadow_descriptor_set, 1, &dynamic_offset);

            vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
    }

    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo begin_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        vkBeginCommandBuffer(cmd, &begin_info);

        // Получаем текущие матрицы из uniform buffer
        SceneUniforms* scene_uni = (SceneUniforms*)scene_uniforms_buffer->mapped_region;

        // --- Проход 1: Рендер карты глубины для направленного света ---
        // Переводим текстуру в режим depth attachment
        transitionShadowImage(cmd, shadow_dir_image,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        renderShadowPass(cmd, shadow_dir_framebuffer, scene_uni->dir_light_matrix);

        // Переводим обратно в режим чтения
        transitionShadowImage(cmd, shadow_dir_image,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // --- Проход 2: Рендер карты глубины для точечного света ---
        transitionShadowImage(cmd, shadow_point_image,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        renderShadowPass(cmd, shadow_point_framebuffer, scene_uni->point_light_matrix);

        transitionShadowImage(cmd, shadow_point_image,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        // --- Проход 3: Основной рендер сцены ---
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