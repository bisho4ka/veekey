#version 450

struct PointLight {
    vec3 position;
    float radius;
    vec3 color;
    float _pad00;
};

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

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

vec3 BlinnPhong(vec3 lightDirection, vec3 normal, vec3 viewDirection, vec3 lightColor) {
    // diffuse component
    float diff = max(dot(normal, lightDirection), 0.0);
    vec3 diffuse = diff * model.albedo_color * lightColor;

    // specular component (Blinn-Phong)
    vec3 halfwayDirection = normalize(lightDirection + viewDirection);
    float spec = pow(max(dot(normal, halfwayDirection), 0.0), model.shininess);
    vec3 specular = spec * model.specular_color * lightColor;

    return diffuse + specular;
}

// attenuation for point light (inverse square law)
float CalculatePointLightAttenuation(float distance, float radius) {
    float t = distance / radius;
    return (1.0 - t) * (1.0 - t); // quadratic attenuation
}

// sunlight by Blinn-Phong
vec3 CalculateSunLight(vec3 normal, vec3 viewDirection) {
    vec3 sunLightDirection = normalize(-scene.sun_light_direction);
    return BlinnPhong(sunLightDirection, normal, viewDirection, scene.sun_light_color);
}

// point light by Blinn-Phong
vec3 CalculatePointLight(PointLight light, vec3 normal, vec3 viewDirection) {
    vec3 lightDirection = normalize(light.position - f_position);
    float distance = length(light.position - f_position);

    if (distance > light.radius) {
        return vec3(0.0);
    }

    float attenuation = CalculatePointLightAttenuation(distance, light.radius);
    vec3 lighting = BlinnPhong(lightDirection, normal, viewDirection, light.color);

    return lighting * attenuation;
}

void main() {
    vec3 normal = normalize(f_normal);
    vec3 viewDirection = normalize(scene.view_position - f_position);

    // ambient component
    vec3 resultLight = model.albedo_color * scene.ambient_light_intensity;

    // sunlight (directional light)
    resultLight += CalculateSunLight(normal, viewDirection);

    // point lights
    for (uint i = 0u; i < scene.point_lights_count; i++) {
        resultLight += CalculatePointLight(point_lights[i], normal, viewDirection);
    }

    final_color = vec4(resultLight, 1.0);
}