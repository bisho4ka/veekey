#version 450

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float _pad00;
};

layout (set = 1, binding = 0) uniform sampler2D u_albedo_texture;

layout(set = 0, binding = 0, std140) uniform SceneUniforms {
    mat4 view_projection;
    mat4 dir_light_matrix;
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
    int material_id;
    uint _pad13[3];
} model;

layout(set = 0, binding = 2, std430) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(set = 0, binding = 3) uniform sampler2DShadow shadowMap;

layout (location = 0) in vec3 f_worldPos;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_dirLightSpacePos;

layout (location = 0) out vec4 final_color;

float calculateShadow(vec4 lightSpacePos) {
    vec3 projCoords = lightSpacePos.xyz / lightSpacePos.w;
    
    // ✅ Преобразуем ТОЛЬКО X и Y из [-1,1] в [0,1]
    projCoords.xy = projCoords.xy * 0.5 + 0.5;
    // Z уже в [0,1] для Vulkan — не трогаем!
    
    // ✅ Добавляем bias для борьбы с self-shadowing
    const float bias = 0.0001;
    projCoords.z -= bias;
    
    // Проверка границ
    if (projCoords.z > 1.0 || projCoords.z < 0.0 || 
        projCoords.x < 0.0 || projCoords.x > 1.0 || 
        projCoords.y < 0.0 || projCoords.y > 1.0) {
        return 1.0; // вне карты теней = нет тени
    }
    
    // ✅ Сэмплируем с учётом bias
    float shadow = texture(shadowMap, projCoords);
    return shadow;
}

vec3 BlinnPhong(vec3 lightDir, vec3 normal, vec3 viewDir, vec3 lightColor, 
                vec3 albedo, vec3 specular, float shininess) {
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo * lightColor;

    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    vec3 specularComp = spec * specular * lightColor;

    return diffuse + specularComp;
}

float CalculatePointLightAttenuation(float distance, float radius) {
    float t = distance / radius;
    return (1.0 - t) * (1.0 - t);
}

vec3 CalculateSunLight(vec3 normal, vec3 viewDir, vec3 albedo, vec3 specular, 
                       float shininess, float shadowFactor) {
    vec3 sunDir = normalize(scene.sun_light_direction);
    return BlinnPhong(sunDir, normal, viewDir, scene.sun_light_color, 
                     albedo, specular, shininess) * shadowFactor;
}

vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 viewDir, 
                         vec3 albedo, vec3 specular, float shininess) {
    vec3 lightDir = normalize(light.position - f_worldPos);
    float distance = length(light.position - f_worldPos);

    if (distance > light.radius) {
        return vec3(0.0);
    }

    float attenuation = CalculatePointLightAttenuation(distance, light.radius);
    return BlinnPhong(lightDir, normal, viewDir, light.color, 
                     albedo, specular, shininess) * attenuation;
}

void main() {
    
    vec4 textureColor = texture(u_albedo_texture, f_uv);
    vec3 albedo = model.albedo_color * textureColor.rgb;
    
    vec3 normal = normalize(f_normal);
    vec3 viewDir = normalize(scene.view_position - f_worldPos);

    float shadowFactor = calculateShadow(f_dirLightSpacePos);
    
    vec3 result = albedo * scene.ambient_light_intensity;
    result += CalculateSunLight(normal, viewDir, albedo, model.specular_color, 
                                model.shininess, shadowFactor);
    
    for (uint i = 0u; i < scene.point_lights_count; i++) {
        result += CalculatePointLight(point_lights[i], normal, viewDir, 
                                     albedo, model.specular_color, model.shininess);
    }
    
    final_color = vec4(result, 1.0);
}