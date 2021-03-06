#pragma once

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "Core/Core.h"

namespace volcano
{

	// Defines several possible options for camera movement. Used as abstraction to stay away from window-system specific input methods
	enum class Camera_Movement : uint8_t
	{
		FORWARD,
		BACKWARD,
		LEFT,
		RIGHT,
		TOP,
		DOWN,
	};

	// Default camera values
	const float YAW = -90.0f;
	const float PITCH = 0.0f;
	const float SPEED = 8.0f;
	const float SENSITIVITY = 0.1f;
	const float ZOOM = 45.0f;

	// An abstract camera class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
	class Camera
	{
	public:
		// Camera Attributes
		glm::vec3 Position;
		glm::vec3 Front;
		glm::vec3 Up;
		glm::vec3 Right;
		glm::vec3 WorldUp;
		// Euler Angles
		float Yaw;
		float Pitch;
		// Camera options
		float MovementSpeed;
		float MouseSensitivity;
		float Zoom;

		// Constructor with vectors
		Camera(glm::vec3 position = coordinate::origin, glm::vec3 up = coordinate::up, float yaw = YAW, float pitch = PITCH) : Front(coordinate::forward), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
		{
			Position = position;
			WorldUp = up;
			Yaw = yaw;
			Pitch = pitch;
			updateCameraVectors();
		}
		// Constructor with scalar values
		Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) : Front(coordinate::forward), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
		{
			Position = glm::vec3(posX, posY, posZ);
			WorldUp = glm::vec3(upX, upY, upZ);
			Yaw = yaw;
			Pitch = pitch;
			updateCameraVectors();
		}

		// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
		glm::mat4 GetViewMatrix()
		{
			return glm::lookAt(Position, Position + Front, Up);
		}

		// Processes input received from any keyboard-like input system. Accepts input parameter in the form of camera defined ENUM (to abstract it from windowing systems)
		void ProcessKeyboard(Camera_Movement direction, float deltaTime)
		{
			float velocity = MovementSpeed * deltaTime;
			if (direction == Camera_Movement::FORWARD)
				Position += Front * velocity;
			else if (direction == Camera_Movement::BACKWARD)
				Position -= Front * velocity;
			else if (direction == Camera_Movement::LEFT)
				Position -= Right * velocity;
			else if (direction == Camera_Movement::RIGHT)
				Position += Right * velocity;
			else if (direction == Camera_Movement::TOP)
				Position += Up * velocity;
			else if (direction == Camera_Movement::DOWN)
				Position -= Up * velocity;
		}

		// Processes input received from a mouse input system. Expects the offset value in both the x and y direction.
		void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true)
		{
			xoffset *= MouseSensitivity;
			yoffset *= MouseSensitivity;

			Yaw += xoffset;
			Pitch += yoffset;

			// Make sure that when pitch is out of bounds, screen doesn't get flipped
			if (constrainPitch)
			{
				if (Pitch > 89.0f)
					Pitch = 89.0f;
				if (Pitch < -89.0f)
					Pitch = -89.0f;
			}

			// Update Front, Right and Up Vectors using the updated Euler angles
			updateCameraVectors();
		}

		// Processes input received from a mouse scroll-wheel event. Only requires input on the vertical wheel-axis
		void ProcessMouseScroll(float yoffset)
		{
			if (Zoom >= 1.0f && Zoom <= 45.0f)
				Zoom -= yoffset;
			if (Zoom <= 1.0f)
				Zoom = 1.0f;
			if (Zoom >= 45.0f)
				Zoom = 45.0f;
		}

	private:
		// Calculates the front vector from the Camera's (updated) Euler Angles
		void updateCameraVectors()
		{
			// Calculate the new Front vector
			glm::vec3 front;
			front.x = sin(glm::radians(Yaw));
			front.y = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
			front.z = sin(glm::radians(Pitch));
			Front = glm::normalize(front);
			// Also re-calculate the Right and Up vector
			Right = glm::normalize(glm::cross(Front, WorldUp));  // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
			Up = glm::normalize(glm::cross(Right, Front));
		}
	};
}