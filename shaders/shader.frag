#version 450

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float _pad00;
};

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;

layout (location = 0) out vec4 final_color;

layout(binding = 0, std140) uniform SceneUniforms {
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

layout(binding = 2, std430) readonly buffer PointLights {
    PointLight point_lights[];
};

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
    float invSquare = 1.0 / (distance * distance * 0.3 + 0.7);
    float smoothFalloff = (1.0 - t) * (1.0 - t);
    return invSquare * smoothFalloff;
}

vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - f_position);
    float distance = length(light.position - f_position);
    
    if (distance > light.radius) return vec3(0.0);
    
    float attenuation = CalculatePointLightAttenuation(distance, light.radius);
    return BlinnPhong(lightDir, normal, viewDir, light.color) * attenuation;
}

vec3 CalculateSunLight(vec3 normal, vec3 viewDir) {
    vec3 sunDir = normalize(-scene.sun_light_direction);
    return BlinnPhong(sunDir, normal, viewDir, scene.sun_light_color);
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 viewDir = normalize(scene.view_position - f_position);
    
    vec3 result = model.albedo_color * scene.ambient_light_intensity;
    result += CalculateSunLight(normal, viewDir);
    
    for (uint i = 0u; i < scene.point_lights_count; i++) {
        result += CalculatePointLight(point_lights[i], normal, viewDir);
    }
    
    final_color = vec4(result, 1.0);
}