#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
  mat4 model;
  mat4 view;
  mat4 proj;
  mat4 padding1;
}
ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTextCoord;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragColor;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec3 fragPos;

vec2 positions[3] = vec2[](vec2(0.0, -0.5), vec2(-0.5, 0.5), vec2(0.5, 0.5));

// out gl_PerVertex
// {
// 	vec4 gl_Position;
// };

void main() {
  gl_Position = ubo.proj * mat4(mat3(ubo.view)) * vec4(inPosition.xyz * 10.0, 1.0);
  //  gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
  fragNormal = normalize((ubo.model * vec4(inNormal, 1.0)).xyz);
  fragColor = inColor;
  fragTexCoord = inTextCoord;
  fragPos = inPosition;
}