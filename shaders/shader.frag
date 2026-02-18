#version 450

// Определяем структуру PointLight (как во 2-й лабе)
struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float _pad00;
};

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

// Массив точечных светов - используем структуру (как во 2-й лабе)
layout(set = 0, binding = 2, std430) readonly buffer PointLights {
    PointLight point_lights[];
};

// Входные данные из вершинного шейдера
layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

// --- Функции освещения (Блинн-Фонг) ---
vec3 BlinnPhong(vec3 lightDir, vec3 normal, vec3 viewDir, vec3 lightColor, vec3 albedo, vec3 specular, float shininess) {
    // Диффузная компонента
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo * lightColor;

    // Спекулярная компонента
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    vec3 specularComp = spec * specular * lightColor;

    return diffuse + specularComp;
}

// Аттенюация для точечных источников
float CalculatePointLightAttenuation(float distance, float radius) {
    float t = distance / radius;
    return (1.0 - t) * (1.0 - t);
}

// Направленный свет
vec3 CalculateSunLight(vec3 normal, vec3 viewDir, vec3 albedo, vec3 specular, float shininess) {
    vec3 sunDir = normalize(-scene.sun_light_direction);
    return BlinnPhong(sunDir, normal, viewDir, scene.sun_light_color, albedo, specular, shininess);
}

// Точечный свет - передаем всю структуру (как во 2-й лабе)
vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 viewDir, vec3 albedo, vec3 specular, float shininess) {
    vec3 lightDir = normalize(light.position - f_position);
    float distance = length(light.position - f_position);

    if (distance > light.radius) {
        return vec3(0.0);
    }

    float attenuation = CalculatePointLightAttenuation(distance, light.radius);
    vec3 lighting = BlinnPhong(lightDir, normal, viewDir, light.color, albedo, specular, shininess);

    return lighting * attenuation;
}

void main() {
    // 1. Цвет из текстуры
    vec4 textureColor = texture(u_albedo_texture, f_uv);
    
    // 2. Комбинируем с albedo_color
    vec3 albedo = model.albedo_color * textureColor.rgb;
    
    // 3. Нормаль и направление взгляда
    vec3 normal = normalize(f_normal);
    vec3 viewDir = normalize(scene.view_position - f_position);

    // 4. Рассеянный свет
    vec3 result = albedo * scene.ambient_light_intensity;

    // 5. Направленный свет
    result += CalculateSunLight(normal, viewDir, albedo, model.specular_color, model.shininess);

    // 6. Точечные источники - используем тот же подход, что во 2-й лабе
    for (uint i = 0u; i < scene.point_lights_count; i++) {
        result += CalculatePointLight(point_lights[i], normal, viewDir, albedo, model.specular_color, model.shininess);
    }

    final_color = vec4(result, 1.0);
}