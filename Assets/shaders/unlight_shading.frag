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

void main() {
  outColor = vec4(texture(texSampler, fragTexCoord).rgb, 1.0);
}