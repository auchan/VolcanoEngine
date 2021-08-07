#pragma once
#include <optional>
#include <array>
#include <string>
#include <vector>

#include "vulkan/vulkan.h"
#include "Runtime/Camera.h"

// forward declares
struct GLFWwindow;

namespace volcano
{
	struct QueueFamilyIndices
	{
		std::optional<uint32_t> graphicsFamily;
		std::optional<uint32_t> presentFamily;

		bool isComplete() const
		{
			return graphicsFamily.has_value() && presentFamily.has_value();
		}
	};

	struct SwapChainSupportDetails
	{
		VkSurfaceCapabilitiesKHR capabilities = {};
		std::vector<VkSurfaceFormatKHR> formats;
		std::vector<VkPresentModeKHR> presentModes;
	};

	struct Vertex
	{
		glm::vec3 pos;
		glm::vec3 normal;
		glm::vec3 color;
		glm::vec2 texCoord;

		bool operator==(const Vertex& other) const
		{
			return pos == other.pos && texCoord == other.texCoord && color == other.color && normal == other.normal;
		}

	
		/**
		 * \brief 在顶点buffer绑定数组中的索引
		 */
		const static uint32_t BINDING_INDEX = 0;

		static VkVertexInputBindingDescription getBindingDescription() {
			VkVertexInputBindingDescription bindingDescription = {};
			bindingDescription.binding = BINDING_INDEX;
			bindingDescription.stride = sizeof(Vertex);
			bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

			return bindingDescription;
		}

		static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
			std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions = {};

			uint32_t location = 0;
			VkVertexInputAttributeDescription* pAttributeDescription = &attributeDescriptions[location];
			pAttributeDescription->binding = BINDING_INDEX;
			pAttributeDescription->location = location;
			pAttributeDescription->format = VK_FORMAT_R32G32B32_SFLOAT;
			pAttributeDescription->offset = offsetof(Vertex, pos);

			pAttributeDescription = &attributeDescriptions[++location];
			pAttributeDescription->binding = BINDING_INDEX;
			pAttributeDescription->location = location;
			pAttributeDescription->format = VK_FORMAT_R32G32B32_SFLOAT;
			pAttributeDescription->offset = offsetof(Vertex, normal);

			pAttributeDescription = &attributeDescriptions[++location];
			pAttributeDescription->binding = BINDING_INDEX;
			pAttributeDescription->location = location;
			pAttributeDescription->format = VK_FORMAT_R32G32B32_SFLOAT;
			pAttributeDescription->offset = offsetof(Vertex, color);

			pAttributeDescription = &attributeDescriptions[++location];
			pAttributeDescription->binding = BINDING_INDEX;
			pAttributeDescription->location = location;
			pAttributeDescription->format = VK_FORMAT_R32G32_SFLOAT;
			pAttributeDescription->offset = offsetof(Vertex, texCoord);

			return attributeDescriptions;
		}
	};

	class VulkanApplication
	{
	public:
		VulkanApplication();

		void run();

	private:
		void initWindow();

		void initVulkan();
		
		void setupImGui();

		void createInstance();

		std::vector<const char*> getRequiredExtensions();

		void setupDebugCallback();

		void createSurface();

		void pickPhysicalDevice();

		bool isDevicesSuitable(VkPhysicalDevice device);

		QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

		bool checkDeviceExtensionSupport(VkPhysicalDevice device);

		SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

		void createLogicDevice();

		void createSwapChain();

		VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

		VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

		VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

		void createImageViews();

		void createRenderPass();

		void createDescriptorSetLayout();

		void createGraphicsPipeline();

		void createGraphicsPipeline2();

		VkShaderModule createShaderModule(const std::vector<char>& code) const;

		VkShaderModule createShaderModuleFromPath(const std::string& filepath) const;

		void createFramebuffers();

		void createCommandPool();

		void createCommandBuffers();

		void createDepthResources();

		VkFormat findDepthFormat();

		VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling,
			VkFormatFeatureFlags features) const;

		bool hasStencilComponent(VkFormat format);

		VkImage createTextureImage(const std::string& texturePath);

		void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage,
			VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);

		void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);

		void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);

		VkImageView createTextureImageView(VkImage textureImage);

		VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags);

		void createTextureSampler();

		struct MeshBatch
		{
			uint32_t startIndex;
			uint32_t count;
			VkImageView imageView;
		};

		std::vector<MeshBatch> meshBatchs;

		// Load a model from file using the ASSIMP model loader and generate all resources required to render the model
		void loadModel(std::string dirname, std::string filename);

		void loadAssets();

		void createVertexBuffer();

		void createIndexBuffer();

		void createUniformBuffer();

		void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer,
			VkDeviceMemory& bufferMemory);

		void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);

		VkCommandBuffer beginSingleTimeCommands();

		void endSingleTimeCommands(VkCommandBuffer commandBuffer);

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

		void createDescriptorPool();

		void createDescriptorSets();

		void updateDescriptorSets(VkImageView imageView);

		void createSyncObjects();

		void recreateSwapChain();

		void mainLoop();

		void updateFrame();

		void updateCommandBuffers();

		void drawFrame();

		void updateUniformBuffer(uint32_t currentImage);

		void cleanup();

		void cleanupImGui();
		
		void cleanupSwapChain();

		//////////////////////////////////////////////////////////////////////////
		// window related
		//////////////////////////////////////////////////////////////////////////

		void processKey(GLFWwindow* window);

		//////////////////////////////////////////////////////////////////////////
		// static methods
		//////////////////////////////////////////////////////////////////////////

		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
			VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData);

		static void frameBufferResizeCallback(GLFWwindow* window, int width, int height);

		static void onKeyEvent(GLFWwindow* window, int key, int scancode, int action, int mods);

		static void mouseCallback(GLFWwindow* window, double xpos, double ypos);

		static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

		static void checkVkResult(VkResult result);

		//////////////////////////////////////////////////////////////////////////
		// variables
		//////////////////////////////////////////////////////////////////////////

		const char* appName = "Volcano Engine";
		
		const uint32_t WIDTH = 800;
		const uint32_t HEIGHT = 600;
		const int MAX_FRAMES_IN_FLIGHT = 2;

		// camera
		Camera camera;
		glm::vec3 viewPosition{};
		glm::vec3 viewDir{};
		double lastX = WIDTH / 2.0f;
		double lastY = HEIGHT / 2.0f;
		bool firstMouse = true;

		// timing
		double deltaTime = 0.0f; // time between current frame and last frame
		double lastFrame = 0.0f;

		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;

		GLFWwindow* window = nullptr;
		VkAllocationCallbacks* pAllocator = nullptr;
		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		VkDevice device = VK_NULL_HANDLE;
		VkQueue graphicsQueue = VK_NULL_HANDLE;
		VkQueue presentQueue = VK_NULL_HANDLE;
		VkSurfaceKHR surface = VK_NULL_HANDLE;

		// 交换链
		VkSwapchainKHR swapChain = VK_NULL_HANDLE;
		std::vector<VkImage> swapChainImages;
		VkFormat swapChainImageFormat;
		VkExtent2D swapChainExtent{};
		std::vector<VkImageView> swapChainImageViews;
		std::vector<VkFramebuffer> swapChainFramebuffers;

		VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
		VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
		std::vector<VkDescriptorSet> descriptorSets;

		VkRenderPass renderPass = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
		VkPipelineLayout pipelineLayout2 = VK_NULL_HANDLE;
		VkPipeline graphicsPipeline{};
		VkPipeline graphicsPipeline2{};

		VkCommandPool commandPool{};
		std::vector<VkCommandBuffer> commandBuffers;

		// 顶点缓冲
		VkBuffer vertexBuffer = VK_NULL_HANDLE;
		VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
		VkBuffer indexBuffer = VK_NULL_HANDLE;
		VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

		// uniform缓冲
		std::vector<VkBuffer> uniformBuffers;
		std::vector<VkDeviceMemory> uniformBuffersMemory;

		std::vector<VkBuffer> lightUniformBuffers;
		std::vector<VkDeviceMemory> lightUniformBuffersMemory;

		// 纹理图像
		//VkImage textureImage;
		//VkDeviceMemory textureImageMemory;
		//VkImageView textureImageView;
		std::vector<VkImage> textureImages;
		std::vector<VkDeviceMemory> textureImageMemories;
		std::vector<VkImageView> textureImageViews;
		VkSampler textureSampler{};

		// 深度图像
		VkImage depthImage{};
		VkDeviceMemory depthImageMemory{};
		VkImageView depthImageView{};

		// 信号量
		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;

		size_t currentFrame = 0;
		bool framebufferResized = false;

		// 校验层
		const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
#ifdef NDEBUG
		const bool enableValidationLayers = false;
#else
		const bool enableValidationLayers = true;
#endif // NDEBUG

		const std::vector<const char*> deviceExtension = {
			VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME
		};

		// ImGui
		VkDescriptorPool imGuiDescriptorPool = VK_NULL_HANDLE;
	};
}
