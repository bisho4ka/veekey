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

layout(set = 0, binding = 2, std430) readonly buffer PointLights {
    PointLight point_lights[];
};

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

vec3 BlinnPhong(vec3 lightDir, vec3 normal, vec3 viewDir, vec3 lightColor, vec3 albedo, vec3 specular, float shininess) {
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * albedo * lightColor;

    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
    vec3 specularComp = spec * specular * lightColor;

    return diffuse + specularComp;
}

float CalculatePointLightAttenuation(float distance, float radius) {
    float t = distance / radius;
    float invSquare = 1.0 / (distance * distance * 0.3 + 0.7);
    float smoothFalloff = (1.0 - t) * (1.0 - t);
    return invSquare * smoothFalloff;
}

vec3 CalculateSunLight(vec3 normal, vec3 viewDir, vec3 albedo, vec3 specular, float shininess) {
    vec3 sunDir = normalize(-scene.sun_light_direction);
    return BlinnPhong(sunDir, normal, viewDir, scene.sun_light_color, albedo, specular, shininess);
}

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
    vec4 textureColor = texture(u_albedo_texture, f_uv);
    
    vec3 albedo = model.albedo_color * textureColor.rgb;
    
    vec3 normal = normalize(f_normal);
    vec3 viewDir = normalize(scene.view_position - f_position);

    vec3 result = albedo * scene.ambient_light_intensity;

    result += CalculateSunLight(normal, viewDir, albedo, model.specular_color, model.shininess);

    for (uint i = 0u; i < scene.point_lights_count; i++) {
        result += CalculatePointLight(point_lights[i], normal, viewDir, albedo, model.specular_color, model.shininess);
    }

    final_color = vec4(result, 1.0);
}