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
  vec3 ambient_light_intensity;
  vec3 sun_light_direction;
  vec3 sun_light_color;
  uint point_lights_count;
  uint spot_lights_count;
};

layout (binding = 1, std140) uniform ModelUniforms {
  mat4 model;
  vec3 albedo_color;
  vec3 specular_color;
  float shininess;
};

void main() {
  vec4 position = model * vec4(v_position, 1.0f);

  mat3 normal_mat = transpose(inverse(mat3(model)));
  vec3 world_normal_mat = normalize(normal_mat * v_normal);

  gl_Position = view_projection * position;

  f_position = position.xyz;
  f_normal = world_normal_mat;
  f_uv = v_uv;
}