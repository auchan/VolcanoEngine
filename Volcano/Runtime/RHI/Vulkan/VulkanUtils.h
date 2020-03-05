#pragma once
#include <vector>
#include <stdexcept>
#include <iostream>

#include "vulkan/vulkan_core.h"

namespace volcano
{
	class VulkanUtils
	{
	public:
		static VkInstance createInstance(const std::vector<const char*>& requiredExtensions, const std::vector<const char*>& validationLayers, bool enableValidationLayers, VkAllocationCallbacks* pAllocator = nullptr)
		{
			if (enableValidationLayers && !checkValidationLayerSupport(validationLayers))
			{
				throw std::runtime_error("validation layers requested, but not available!");
			}

			VkApplicationInfo appInfo = createApplicationInfo();
			VkInstanceCreateInfo createInfo = createInstanceCreateInfo(appInfo);

			// 打印必需的拓展
			std::cout << "required extension:" << std::endl;
			for (const char* extensionName : requiredExtensions)
			{
				std::cout << "\t" << extensionName << std::endl;
			}

			// 检测instance支持的拓展
			uint32_t extensionCount = 0;
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
			std::vector<VkExtensionProperties> extensions(extensionCount);
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
			
			// 打印instance支持的拓展
			std::cout << "available extension:" << std::endl;
			for (const auto& extension : extensions)
			{
				std::cout << "\t" << extension.extensionName << std::endl;
			}

			createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
			createInfo.ppEnabledExtensionNames = requiredExtensions.data();

			// 设置校验层
			if (enableValidationLayers)
			{
				createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();
			}
			else
			{
				// 全局校验层设置为0，表示不使用校验层
				createInfo.enabledLayerCount = 0;
			}

			VkInstance instance = VK_NULL_HANDLE;
			if (vkCreateInstance(&createInfo, pAllocator, &instance) != VK_SUCCESS)
			{
				throw std::runtime_error("failed to create instance!");
			}
			return instance;
		}

		static bool checkValidationLayerSupport(const std::vector<const char*>& validationLayers)
		{
			uint32_t layerCount;
			vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

			std::vector<VkLayerProperties> availableLayers(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

			for (const char* layerName : validationLayers)
			{
				bool layerFound = false;
				for (const auto& layerProperties : availableLayers)
				{
					if (strcmp(layerName, layerProperties.layerName) == 0)
					{
						layerFound = true;
						break;
					}
				}

				if (!layerFound)
				{
					return false;
				}
			}

			return true;
		}

		static VkApplicationInfo createApplicationInfo()
		{
			VkApplicationInfo appInfo = {};
			appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			appInfo.pApplicationName = "Volcano Application";
			appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.pEngineName = "Volcano Engine";
			appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.apiVersion = VK_API_VERSION_1_1;
			return appInfo;
		}

		static VkInstanceCreateInfo createInstanceCreateInfo(VkApplicationInfo& appInfo)
		{
			VkInstanceCreateInfo createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.pApplicationInfo = &appInfo;
			return createInfo;
		}

		static VkResult createDebugUtilsMessengerEXT(
			VkInstance instance,
			const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
			const VkAllocationCallbacks* pAllocator,
			VkDebugUtilsMessengerEXT* pDebugMessenger)
		{
			const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
				instance, "vkCreateDebugUtilsMessengerEXT"));
			if (func == nullptr)
			{
				return VK_ERROR_EXTENSION_NOT_PRESENT;
			}
			return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
		}

		static void destroyDebugUtilsMessengerEXT(
			VkInstance instance,
			VkDebugUtilsMessengerEXT pDebugMessenger,
			const VkAllocationCallbacks* pAllocator)
		{
			const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
				instance, "vkDestroyDebugUtilsMessengerEXT"));
			if (func == nullptr)
			{
				return;
			}
			func(instance, pDebugMessenger, pAllocator);
		}
	};
}

