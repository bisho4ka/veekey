#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;
layout (location = 3) out vec4 f_dirLightSpacePos;
layout (location = 4) out vec4 f_pointLightSpacePos;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    float _pad0;
    vec3 ambient_light_intensity;
    float _pad1;
    vec3 sun_light_direction;
    float _pad2;
    vec3 sun_light_color;
    float _pad3;
    mat4 dir_light_matrix;
    mat4 point_light_matrix;
    uint point_lights_count;
    uint _pad4[3];
} scene;

layout(set = 0, binding = 1, std140) uniform ModelUniforms {
    mat4 model;
    vec3 albedo_color;
    float _pad10;
    vec3 specular_color;
    float _pad12;
    float shininess;
    uint _pad13[3];
} model;

void main() {
    vec4 world_pos = model.model * vec4(v_position, 1.0);
    gl_Position = scene.view_projection * world_pos;

    f_position = world_pos.xyz;
    
    mat3 normalMatrix = transpose(inverse(mat3(model.model)));
    f_normal = normalize(normalMatrix * v_normal);
    
    f_uv = v_uv;

    // Координаты в пространстве источников света для теней
    f_dirLightSpacePos = scene.dir_light_matrix * world_pos;
    f_pointLightSpacePos = scene.point_light_matrix * world_pos;
}