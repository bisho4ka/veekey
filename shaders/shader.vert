#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;

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
    uint _pad4[3];
} scene;

layout (binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad10;
    vec3 specular_color;
    float _pad12;
    float shininess;
    uint _pad13[3];
} model;

void main() {
    vec4 position = model.model * vec4(v_position, 1.0f);
    
    mat3 normal_mat = transpose(inverse(mat3(model.model)));
    vec3 world_normal_mat = normalize(normal_mat * v_normal);
    
    gl_Position = scene.view_projection * position;
    
    f_position = position.xyz;
    f_normal = world_normal_mat;
}