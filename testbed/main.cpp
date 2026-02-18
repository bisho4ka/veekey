#include <cstdint>
#include <climits>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include <veekay/veekay.hpp>

#include <imgui.h>
#include <vulkan/vulkan_core.h>

namespace {
    //поле зрения камеры
    constexpr float camera_fov = 70.0f;
    //ближняя площадь отсечения
    constexpr float camera_near_plane = 0.01f;
    //дальняя площадь отсечения
    constexpr float camera_far_plane = 100.0f;

    //вращение всей системы
    float world_rotation = 0.0f;

    //наклон родительского объекта
    float big_pyramid_tilt = 1.2f;

    //масштаб дочернего объекта
    float little_pyramid_scale = 0.3f;

    //радиус орбиты дочернего объекта
    float little_orbit_radius = 3.0f;


    struct Matrix {
        float m[4][4];
    };

    struct Vector {
        float x, y, z;
    };

    struct Vertex {
        Vector position;
        Vector color;
    };

    struct ShaderConstants {
        Matrix projection;
        Matrix transform;
        Vector color;
    };

    struct VulkanBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
    };

    VkShaderModule vertex_shader_module;
    VkShaderModule fragment_shader_module;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;

    VulkanBuffer vertex_buffer;
    VulkanBuffer index_buffer;

    Vector model_position = {-1.0f, 1.0f, 5.0f};
    float model_rotation;

    //функция получения единичной матрицы
    Matrix identity() {
        Matrix result{};

        result.m[0][0] = 1.0f;
        result.m[1][1] = 1.0f;
        result.m[2][2] = 1.0f;
        result.m[3][3] = 1.0f;

        return result;
    }

    //функция получения матрицы перспективной проекции
    Matrix projection(float fov, float aspect_ratio, float near, float far) {
        Matrix result{};

        const float radians = fov * M_PI / 180.0f;
        const float cot = 1.0f / tanf(radians / 2.0f);

        result.m[0][0] = cot / aspect_ratio;
        result.m[1][1] = cot;
        result.m[2][3] = 1.0f;

        result.m[2][2] = far / (far - near);
        result.m[3][2] = (-near * far) / (far - near);

        return result;
    }


    //матрица перемещения
    Matrix translation(Vector vector) {
        Matrix result = identity();

        result.m[3][0] = vector.x;
        result.m[3][1] = vector.y;
        result.m[3][2] = vector.z;

        return result;
    }

    //матрица поворота
    Matrix rotation(Vector axis, float angle) {
        Matrix result{};

        float length = sqrtf(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);

        axis.x /= length;
        axis.y /= length;
        axis.z /= length;

        float sina = sinf(angle);
        float cosa = cosf(angle);
        float cosv = 1.0f - cosa;

        result.m[0][0] = (axis.x * axis.x * cosv) + cosa;
        result.m[0][1] = (axis.x * axis.y * cosv) + (axis.z * sina);
        result.m[0][2] = (axis.x * axis.z * cosv) - (axis.y * sina);

        result.m[1][0] = (axis.y * axis.x * cosv) - (axis.z * sina);
        result.m[1][1] = (axis.y * axis.y * cosv) + cosa;
        result.m[1][2] = (axis.y * axis.z * cosv) + (axis.x * sina);

        result.m[2][0] = (axis.z * axis.x * cosv) + (axis.y * sina);
        result.m[2][1] = (axis.z * axis.y * cosv) - (axis.x * sina);
        result.m[2][2] = (axis.z * axis.z * cosv) + cosa;

        result.m[3][3] = 1.0f;

        return result;
    }

    //умножение матриц
    Matrix multiply(const Matrix &a, const Matrix &b) {
        Matrix result{};

        for (int j = 0; j < 4; j++) {
            for (int i = 0; i < 4; i++) {
                for (int k = 0; k < 4; k++) {
                    result.m[j][i] += a.m[j][k] * b.m[k][i];
                }
            }
        }

        return result;
    }

    //функция, которая загружает шейдерные модули
    //принимает уже скомпилированный код шейдера и возвращает дескриптор шейдерного модуля
    VkShaderModule loadShaderModule(const char *path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        size_t size = file.tellg();
        std::vector<uint32_t> buffer(size / sizeof(uint32_t));
        file.seekg(0);
        file.read(reinterpret_cast<char *>(buffer.data()), size);
        file.close();

        VkShaderModuleCreateInfo info{
                .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                .codeSize = size, //размер шейдерного кода в байтах
                .pCode = buffer.data(), //указатель на шейдерный код
        };

        VkShaderModule result;
        //создает шейдерный модуль с помощью API Vulkan
        if (vkCreateShaderModule(veekay::app.vk_device, &
                info, nullptr, &result) != VK_SUCCESS) {
            return nullptr;
        }

        return result;
    }

    //функция для создания буфера (буфер хранит какие-либо данные на GPU)
    //принимает на вход размер буфера, указатель на данные, флаги использования (для вершин или индексов и т.д.)
    VulkanBuffer createBuffer(size_t size, void *data, VkBufferUsageFlags usage) {
        VkDevice &device = veekay::app.vk_device;
        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;

        VulkanBuffer result{};
        {
            VkBufferCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                    .size = size,
                    .usage = usage,
                    .sharingMode = VK_SHARING_MODE_EXCLUSIVE, //буфер будет использоваться только 1 очередью
            };

            if (vkCreateBuffer(device, &info, nullptr, &result.buffer) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan buffer\n";
                return {};
            }
        }


        {
            VkMemoryRequirements requirements;
            //запрашивают требования к памяти для созданного буфера
            vkGetBufferMemoryRequirements(device, result.buffer, &requirements);

            VkPhysicalDeviceMemoryProperties properties;
            //получает информацию о доступных типах памяти на GPU
            vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);

            //может писать и читать в нее, автоматически синхронизируется с CPU
            const VkMemoryPropertyFlags flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

            uint32_t index = UINT_MAX;
            for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
                const VkMemoryType &type = properties.memoryTypes[i];

                if ((requirements.memoryTypeBits & (1 << i)) &&
                    (type.propertyFlags & flags) == flags) {
                    index = i;
                    break;
                }
            }

            if (index == UINT_MAX) {
                std::cerr << "Failed to find required memory type to allocate Vulkan buffer\n";
                return {};
            }

            VkMemoryAllocateInfo info{
                    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    .allocationSize = requirements.size,
                    .memoryTypeIndex = index,
            };

            //выделяет память для буфера
            if (vkAllocateMemory(device, &info, nullptr, &result.memory) != VK_SUCCESS) {
                std::cerr << "Failed to allocate Vulkan buffer memory\n";
                return {};
            }

            //связывает аллоцированную память с буфером
            if (vkBindBufferMemory(device, result.buffer, result.memory, 0) != VK_SUCCESS) {
                std::cerr << "Failed to bind Vulkan  buffer memory\n";
                return {};
            }

            void *device_data;
            //отображает память на GPU в адресное пространство CPU
            vkMapMemory(device, result.memory, 0, requirements.size, 0, &device_data);
            //копирует данные из CPU в GPU
            memcpy(device_data, data, size);
            //завершает отображение
            vkUnmapMemory(device, result.memory);
        }

        return result;
    }

    //функция уничтожения буфера
    void destroyBuffer(const VulkanBuffer &buffer) {
        VkDevice &device = veekay::app.vk_device;

        //освобождает память, выделенную для GPU
        vkFreeMemory(device, buffer.memory, nullptr);
        //уничтожает логический объект буфера
        vkDestroyBuffer(device, buffer.buffer, nullptr);
    }

    //функция для инициализации графического пайплайна
    void initialize(VkCommandBuffer cmd) {
        VkDevice &device = veekay::app.vk_device;

        VkPhysicalDevice &physical_device = veekay::app.vk_physical_device;
        {
            //загрузка уже скомпилированного вершинного шейдера (этот шейдер преобразует вершины)
            vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
            if (!vertex_shader_module) {
                std::cerr << "Failed to load Vulkan vertex shader from file\n";
                veekay::app.running = false;
                return;
            }

            //загрузка уже скомпилированного фрагментного шейдера (этот шейдер вычисляет цвет пикселей)
            fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
            if (!fragment_shader_module) {
                std::cerr << "Failed to load Vulkan fragment shader from file\n";
                veekay::app.running = false;
                return;
            }

            //настройка шейдерных стадий
            VkPipelineShaderStageCreateInfo stage_infos[2];

            //указываем, в каком порядке используем шейдеры
            //сначала используем вершинный шейдер
            stage_infos[0] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_VERTEX_BIT,
                    .module = vertex_shader_module,
                    .pName = "main",
            };

            //потом используем фрагментный шейдер
            stage_infos[1] = VkPipelineShaderStageCreateInfo{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
                    .module = fragment_shader_module,
                    .pName = "main",
            };

            //описываем, как данные вершин упакованы в буфере
            VkVertexInputBindingDescription buffer_binding{
                    .binding = 0,
                    .stride = sizeof(Vertex), //расстояние между вершинами в байтах
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            };

            //описание атрибутов вершины (первые 3 компонента - позиция, вторые три компонента - цвет)
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
                            .offset = offsetof(Vertex, color),
                    },
            };

            //описание формата вершин
            VkPipelineVertexInputStateCreateInfo input_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    .vertexBindingDescriptionCount = 1,
                    .pVertexBindingDescriptions = &buffer_binding,
                    .vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
                    .pVertexAttributeDescriptions = attributes,
            };

            //описание состояний графического конвеера
            VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    //примитивами являются отдельные треугольники
                    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            };

            //описание правил растеризации
            VkPipelineRasterizationStateCreateInfo raster_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    .polygonMode = VK_POLYGON_MODE_FILL,
                    //отсекаем задние грани
                    .cullMode = VK_CULL_MODE_BACK_BIT,
                    //лицевая сторона определяется по часовой стрелке
                    .frontFace = VK_FRONT_FACE_CLOCKWISE,
                    .lineWidth = 1.0f,
            };

            //настройка мультисэмплинга (сглаживания) - отключен
            VkPipelineMultisampleStateCreateInfo sample_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
                    .sampleShadingEnable = false,
                    .minSampleShading = 1.0f,
            };

            //настройка вьюпорта
            VkViewport viewport{
                    .x = 0.0f,
                    .y = 0.0f,
                    .width = static_cast<float>(veekay::app.window_width),
                    .height = static_cast<float>(veekay::app.window_height),
                    .minDepth = 0.0f,
                    .maxDepth = 1.0f,
            };

            //настройка ножниц - прямоугольника отсечения (отключены)
            VkRect2D scissor{
                    .offset = {0, 0},
                    .extent = {veekay::app.window_width, veekay::app.window_height},
            };

            //структура, объединяющая информацию о вьюпорте
            VkPipelineViewportStateCreateInfo viewport_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

                    .viewportCount = 1,
                    .pViewports = &viewport,

                    .scissorCount = 1,
                    .pScissors = &scissor,
            };

            //структура, управляющая обработкой глубины
            VkPipelineDepthStencilStateCreateInfo depth_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    //включение теста глубины
                    .depthTestEnable = true,
                    //разрешение записи в буфер глубины
                    .depthWriteEnable = true,
                    //оператор сравнения (проходит, если глубина <= текущей)
                    .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };

            //настройка смешивания цветов (отключена)
            VkPipelineColorBlendAttachmentState attachment_info{
                    .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                      VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT |
                                      VK_COLOR_COMPONENT_A_BIT,
            };

            //настройка глобального состояния смешивания цветов
            VkPipelineColorBlendStateCreateInfo blend_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

                    //логические операции отключены
                    .logicOpEnable = false,
                    .logicOp = VK_LOGIC_OP_COPY,

                    //количество цветовых вложений
                    .attachmentCount = 1,
                    //массив состояний смешивания
                    .pAttachments = &attachment_info
            };

            //настройка диапазона пуш-констант
            VkPushConstantRange push_constants
                    {
                            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                                          VK_SHADER_STAGE_FRAGMENT_BIT,
                            .size = sizeof(ShaderConstants),
                    };

            //создание макета пайплайна (какие ресурсы будут доступны шейдерам ?)
            VkPipelineLayoutCreateInfo layout_info{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .pushConstantRangeCount = 1,
                    .pPushConstantRanges = &push_constants,
            };

            //создание макета пайплайна
            if (vkCreatePipelineLayout(device, &layout_info,
                                       nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline layout\n";
                veekay::app.running = false;
                return;
            }

            //структура, совмещающая всю информацию о пайплайне
            VkGraphicsPipelineCreateInfo info{
                    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    //количество шейдерных стадий
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

            //создание графического пайплайна
            if (vkCreateGraphicsPipelines(device, nullptr,
                                          1, &info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create Vulkan pipeline\n";
                veekay::app.running = false;
                return;
            }
        }

        //создание 3d модели пирамиды - теперь все вершины белые, цвет будем задавать через push constants
        Vertex vertices[] = {
                {{-1.0f, -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},  // белый
                {{1.0f,  -1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},  // белый
                {{1.0f,  1.0f,  0.0f}, {1.0f, 0.0f, 0.0f}},  // белый
                {{-1.0f, 1.0f,  0.0f}, {1.0f, 0.0f, 0.0f}},  // белый
                {{0.0f,  0.0f,  2.0f}, {1.0f, 0.0f, 0.0f}}   // белый
        };

        //определения треугольников пирамиды
        uint32_t indices[] = {
                //первый треугольник основания
                0, 1, 2,
                //второй треугольник основания
                2, 3, 0,

                //передняя грань
                0, 1, 4,
                //правая грань
                1, 2, 4,
                //задняя грань
                2, 3, 4,
                //левая грань
                3, 0, 4,
        };

        //создание вершинного буфера на GPU
        vertex_buffer = createBuffer(sizeof(vertices), vertices,
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

        //создание индексного буфера на GPU
        index_buffer = createBuffer(sizeof(indices), indices,
                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }

    //функция очистки ресурсов
    void shutdown() {
        VkDevice &device = veekay::app.vk_device;

        destroyBuffer(index_buffer);
        destroyBuffer(vertex_buffer);

        //уничтожение графического пайплайна
        vkDestroyPipeline(device, pipeline, nullptr);
        //уничтожение макета графического пайплайна
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyShaderModule(device, fragment_shader_module, nullptr);
        vkDestroyShaderModule(device, vertex_shader_module, nullptr);
    }

    //функция обновления состояния приложения
    void update(double time) {
        //окно управления
        ImGui::Begin("Control");

        ImGui::Separator();
        ImGui::Text("Little pyramid");
        
        //слайдер для масштаба дочернего объекта
        ImGui::SliderFloat("Scale", &little_pyramid_scale, 0.1f, 2.0f);
        
        //слайдер для радиуса орбиты дочернего объекта
        ImGui::SliderFloat("Radius", &little_orbit_radius, 0.0f, 5.0f);

        ImGui::End();

        //сколько времени прошло с предыдущего кадра
        static double last_time = time;
        double delta_time = time - last_time;
        last_time = time;
        float world_rotation_speed = 0.5f;
        float speed = 1.0f;

        //накопление общего времени анимации
        static double total_time = 0.0;

        // АНИМАЦИЯ ВСЕГДА АКТИВНА (убрана проверка animation_stopped)
        total_time += delta_time;  // Убрано movement_direction

        model_rotation = float(total_time) * speed;
        world_rotation = float(total_time) * world_rotation_speed;

        //защита от переполнения
        const double max_time = 50000.0;
        if (fabs(total_time) > max_time) {
            total_time = fmod(total_time, max_time);
        }
    }

    //функция рендеринга - записи всех команд для отрисовки кадра
    void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
        //подготовка командного буфера
        vkResetCommandBuffer(cmd, 0);
        {

            VkCommandBufferBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    //буфер будет использоваться только 1 раз
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            };

            vkBeginCommandBuffer(cmd, &info);
        }
        {

            //цвет очистки
            VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
            //глубина очистки
            VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

            VkClearValue clear_values[] = {clear_color, clear_depth};

            //параметры начала прохода рендеринга
            VkRenderPassBeginInfo info{
                    .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                    .renderPass = veekay::app.vk_render_pass,
                    //фреймбуфер - набор конкретных изображений, в которые производится рендеринг
                    .framebuffer = framebuffer,
                    .renderArea = {
                            .extent = {
                                    veekay::app.window_width,
                                    veekay::app.window_height
                            },
                    },
                    .clearValueCount = 2,
                    //массив значений очистки
                    .pClearValues = clear_values,
            };

            //начало прохода
            vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
        }


        {
            //привязываем графический пайплайн
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);


            //привязывает вершинный и индексный буферы
            VkDeviceSize offset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer.buffer, &offset);
            vkCmdBindIndexBuffer(cmd, index_buffer.buffer, offset, VK_INDEX_TYPE_UINT32);


            //создает матрицу перспективной проекции, учитывая соотношения окна
            Matrix m_proj = projection(
                    camera_fov,
                    float(veekay::app.window_width) / float(veekay::app.window_height),
                    camera_near_plane, camera_far_plane);


            //стек матриц для иерархических преобразований
            std::vector<Matrix> stack;

            stack.push_back(identity());

            //создание матрицы вращений относительно оси Z
            Matrix system_matrix = rotation({0.0f, 0.0f, 1.0f}, world_rotation);

            //применяем к матрице на вершине стека
            stack.push_back(multiply(stack.back(), system_matrix));

            //применяем преобразования к родительской матрице: наклон, перемещение
            Matrix big_rotation_matrix = rotation({0.0f, 1.0f, 0.0f}, big_pyramid_tilt);

            Matrix big_local_transf = translation(model_position);

            big_local_transf = multiply(big_rotation_matrix, big_local_transf);

            Matrix world_big = multiply(big_local_transf, stack.back());

            //создаем пуш-константы для родительской пирамиды
            ShaderConstants big_constants{
                    .projection = m_proj,
                    .transform = world_big,
                    .color = {1.0f, 0.0f, 0.0f},
            };

            //передаем пуш-константы родительской пирамиды
            vkCmdPushConstants(cmd, pipeline_layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(ShaderConstants), &big_constants);

            //отрисовываем родительскую пирамиду
            vkCmdDrawIndexed(cmd, 18, 1, 0, 0, 0);

            //сохраняем родительскую пирамиду для дочерней
            stack.push_back(world_big);

            //создаем матрицу для дочерней пирамиды
            Matrix little_pyramid = identity();

            for (int i = 0; i < 3; i++) {
                little_pyramid.m[i][i] = little_pyramid_scale;
            }

            float orbit_angle_little = model_rotation * 1.5f;

            Matrix little_orbit_transl = translation({
                                                             cosf(orbit_angle_little) * little_orbit_radius,
                                                             sinf(orbit_angle_little) * little_orbit_radius,
                                                             0.0f
                                                     });

            //комбинируем матрицу масштаба, матрицу орбиты и родительскую
            Matrix little_local_transf = multiply(little_pyramid, little_orbit_transl);

            Matrix world_little = multiply(little_local_transf, stack.back());

            //создаем пуш-константы для дочерней матрицы
            ShaderConstants little_pyramid_constants{
                    .projection = m_proj,
                    .transform = world_little,
            };

            vkCmdPushConstants(cmd, pipeline_layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(ShaderConstants), &little_pyramid_constants);
            vkCmdDrawIndexed(cmd, 18, 1, 0, 0, 0);

            stack.pop_back();
            stack.pop_back();
        }
        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }
}

int main() {
    return veekay::run({
        .init = initialize,
        .shutdown = shutdown,
        .update = update,
        .render = render,
    });
}