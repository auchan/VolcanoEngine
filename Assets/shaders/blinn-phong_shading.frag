#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 1) uniform sampler2D texSampler;
layout(binding = 2, std140) uniform Light {
  vec3 pos;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
}
light;

layout(push_constant) uniform PushConstant { vec3 viewPos; }
pushConstant;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec3 fragColor;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

vec3 lightColor = vec3(1.0f, 1.0f, 1.0f);
vec3 lightPos = vec3(0.0f, 0.0f, 0.0f);

struct Material {
  vec3 ambient;
  vec3 diffuse;
  float specular;
  float shininess;
} material;

void main() {
  material.ambient = vec3(0.0f, 0.0f, 0.0f);
  material.specular = 0.5f;
  material.shininess = 5.f;
  material.diffuse = texture(texSampler, fragTexCoord).rgb;

  lightPos = light.pos;
  lightColor = light.ambient;
  vec3 Normal = fragNormal;
  vec3 FragPos = fragPos;
  vec3 viewPos = pushConstant.viewPos;

  // 环境光
  vec3 ambient = lightColor * material.ambient;

  // 漫反射
  vec3 norm = normalize(Normal);
  vec3 lightDir = normalize(lightPos - FragPos);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = lightColor * (diff * material.diffuse);

  // 镜面光
  vec3 viewDir = normalize(viewPos - FragPos);
  vec3 halfwayDir = normalize(lightDir + viewDir);
  float spec = pow(max(dot(norm, halfwayDir), 0), material.shininess);
  vec3 specular = lightColor * (spec * material.specular);

  vec3 result = ambient + diffuse + specular;
  outColor = vec4(result, 1.0);
}