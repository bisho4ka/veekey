#version 450

// --- Текстуры (Material Set 1) ---
layout (set = 1, binding = 0) uniform sampler2D u_albedo_texture;

// --- Данные сцены (Scene Set 0) ---
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

// Точечные света - ИСПРАВЛЕНО: правильный синтаксис для массива структур
layout(set = 0, binding = 2, std430) readonly buffer PointLights {
    vec3 position;
    float radius;
    vec3 color;
    float _pad0;
} point_lights[];

// --- Входные данные из вершинного шейдера ---
layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

// --- Вспомогательные функции ---
vec3 BlinnPhong(vec3 lightDir, vec3 normal, vec3 viewDir, vec3 lightColor) {
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * model.albedo_color * lightColor;

    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), model.shininess);
    vec3 specular = spec * model.specular_color * lightColor;

    return diffuse + specular;
}

float CalculatePointLightAttenuation(float distance, float radius) {
    float t = distance / radius;
    return (1.0 - t) * (1.0 - t);
}

// --- Главная функция ---
void main() {
    // 1. Цвет из текстуры
    vec4 textureColor = texture(u_albedo_texture, f_uv);
    
    // 2. Нормаль и направление взгляда
    vec3 normal = normalize(f_normal);
    vec3 viewDir = normalize(scene.view_position - f_position);

    // 3. Базовый рассеянный свет
    vec3 result = textureColor.rgb * scene.ambient_light_intensity;

    // 4. Направленный свет
    vec3 sunDir = normalize(-scene.sun_light_direction);
    result += BlinnPhong(sunDir, normal, viewDir, scene.sun_light_color) * textureColor.rgb;

    // 5. Точечные света
    for (uint i = 0u; i < scene.point_lights_count; i++) {
        vec3 lightPos = point_lights[i].position;
        float radius = point_lights[i].radius;
        vec3 lightColor = point_lights[i].color;

        vec3 lightDir = normalize(lightPos - f_position);
        float distance = length(lightPos - f_position);

        if (distance < radius) {
            float attenuation = CalculatePointLightAttenuation(distance, radius);
            result += BlinnPhong(lightDir, normal, viewDir, lightColor) * textureColor.rgb * attenuation;
        }
    }

    final_color = vec4(result, 1.0);
}