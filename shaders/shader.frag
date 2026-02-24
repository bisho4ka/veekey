#version 450

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float ambient_intensity;  // Добавлено для управления рассеянным светом
};

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout(binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    vec3 view_position;
    float _pad0;
    vec3 ambient_light_intensity;  // Глобальный рассеянный свет
    float _pad1;
    vec3 sun_light_direction;      // Направление солнечного света
    float _pad2;
    vec3 sun_light_color;          // Цвет солнечного света
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

layout(binding = 2, std430) readonly buffer PointLights {
    PointLight point_lights[];
};

// Закон обратных квадратов с мягким затуханием на границе радиуса
float CalculatePointLightAttenuation(float distance, float radius) {
    // Закон обратных квадратов (inverse square law)
    // +0.01 чтобы избежать деления на ноль
    float attenuation = 1.0 / (distance * distance + 0.01);
    
    // Плавное затухание до 0 на границе радиуса
    float t = clamp(1.0 - distance / radius, 0.0, 1.0);
    
    return attenuation * t;
}

vec3 BlinnPhong(vec3 lightDir, vec3 normal, vec3 viewDir, vec3 lightColor, float ambientIntensity) {
    // Нормализуем направление к источнику света
    vec3 normLightDir = normalize(lightDir);
    
    // Ambient компонент для этого источника
    vec3 ambient = model.albedo_color * lightColor * ambientIntensity;
    
    // Diffuse компонент (Ламберт)
    float diff = max(dot(normal, normLightDir), 0.0);
    vec3 diffuse = diff * model.albedo_color * lightColor;
    
    // Specular компонент (Blinn-Phong)
    vec3 halfwayDir = normalize(normLightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), model.shininess);
    vec3 specular = spec * model.specular_color * lightColor;
    
    return ambient + diffuse + specular;
}

// Расчет направленного света (солнца)
vec3 CalculateSunLight(vec3 normal, vec3 viewDir) {
    // Направление от поверхности к солнцу (противоположно направлению света)
    vec3 lightDir = normalize(-scene.sun_light_direction);
    
    // Для направленного света используем фиксированную ambient интенсивность 0.1
    return BlinnPhong(lightDir, normal, viewDir, scene.sun_light_color, 0.1);
}

// Расчет точечного источника света
vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 viewDir) {
    // Вектор от поверхности к источнику света
    vec3 lightDir = light.position - f_position;
    float distance = length(lightDir);
    
    // Проверка радиуса действия
    if (distance > light.radius) {
        return vec3(0.0);
    }
    
    // Закон обратных квадратов
    float attenuation = CalculatePointLightAttenuation(distance, light.radius);
    
    // Расчет освещения по Блинн-Фонгу с учетом ambient для этого источника
    vec3 lighting = BlinnPhong(lightDir, normal, viewDir, light.color, light.ambient_intensity);
    
    return lighting * attenuation;
}

void main() {
    // Нормализуем нормаль и направление обзора
    vec3 normal = normalize(f_normal);
    vec3 viewDir = normalize(scene.view_position - f_position);
    
    // Глобальный ambient свет (окружение)
    vec3 result = model.albedo_color * scene.ambient_light_intensity;
    
    // Добавляем направленный свет (солнце)
    result += CalculateSunLight(normal, viewDir);
    
    // Добавляем все точечные источники света
    for (uint i = 0u; i < scene.point_lights_count; i++) {
        result += CalculatePointLight(point_lights[i], normal, viewDir);
    }
    
    final_color = vec4(result, 1.0);
}