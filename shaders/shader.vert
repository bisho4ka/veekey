#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

layout (binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    float _pad0;
    vec3 ambient_light_intensity;
    float _pad1;
    vec3 sun_light_direction;
    float _pad2;
    vec3 sun_light_color;
    float _pad3;
    uint point_lights_count;
    uint _pad4[3];  // Выравнивание до 16 байт
};

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad10;
    vec3 specular_color;
    float _pad12;
    float shininess;
    uint _pad13[3];
};

void main() {
    // Перевод вершины в мировые координаты
    vec4 world_position = model * vec4(v_position, 1.0);
    
    // Матрица нормалей для преобразования нормалей в мировые координаты
    // (транспонированная обратная матрица модели)
    mat3 normal_matrix = transpose(inverse(mat3(model)));
    vec3 world_normal = normalize(normal_matrix * v_normal);
    
    // Итоговая позиция в clip space
    gl_Position = view_projection * world_position;
    
    // Передача данных во фрагментный шейдер
    f_position = world_position.xyz;
    f_normal = world_normal;
    f_uv = v_uv;
}