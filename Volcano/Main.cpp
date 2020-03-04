#include <iostream>
#include <Runtime/RHI/Vulkan/VulkanApplication.h>

int main()
{	
	volcano::VulkanApplication app;
	
	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}