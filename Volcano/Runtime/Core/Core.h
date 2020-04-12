#pragma once
#include <glm/vec3.hpp>

namespace volcano::vec3
{	
	static const glm::vec3 zero = glm::vec3(0, 0, 0);

	static const glm::vec3 one = glm::vec3(1, 1, 1);
}

/**
 * \brief 右手坐标系：
 * x -> 右
 * y -> 前
 * z -> 上
 */
namespace volcano::coordinate
{
	static const glm::vec3 origin = glm::vec3(0, 0, 0);
	
	static const glm::vec3 forward = glm::vec3(0, 1, 0);

	static const glm::vec3 backward = glm::vec3(0, -1, 1);

	static const glm::vec3 right = glm::vec3(1, 0, 0);

	static const glm::vec3 left = glm::vec3(-1, 0, 0);
	
	static const glm::vec3 up = glm::vec3(0, 0, 1);

	static const glm::vec3 down = glm::vec3(0, 0, -1);
}