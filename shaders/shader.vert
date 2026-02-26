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
    float _pad0;                    // ВАЖНО!
    vec3 ambient_light_intensity;
    float _pad1;                     // ВАЖНО!
    vec3 sun_light_direction;
    float _pad2;                     // ВАЖНО!
    vec3 sun_light_color;
    float _pad3;                     // ВАЖНО!
    uint point_lights_count;
    uint _pad4[3];                   // ВАЖНО!
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad10;                    // Из вашей структуры!
    vec3 specular_color;
    float _pad12;                    // Из вашей структуры!
    float shininess;
    uint _pad13[3];                  // Из вашей структуры!
} model;

void main() {
    vec4 position = model.model * vec4(v_position, 1.0f);  // model.model, а не просто model!
    
    mat3 normal_mat = transpose(inverse(mat3(model.model)));
    vec3 world_normal_mat = normalize(normal_mat * v_normal);
    
    gl_Position = scene.view_projection * position;
    
    f_position = position.xyz;
    f_normal = world_normal_mat;
    f_uv = v_uv;  // UV не нужны для 2 лабы, но можно оставить
}