#include "AssetManager.h"
#include <fstream>

using namespace volcano;

std::string AssetManager::getAssetPath(const std::string &relativePath)
{
	return std::string("..\\Assets\\") + relativePath;
}

std::vector<char> AssetManager::readFile(const std::string& filepath)
{
	const std::string& absFilepath = AssetManager::getAssetPath(filepath);
	// ate 从文件尾部开始读取
	std::ifstream file(absFilepath, std::ios::ate | std::ios::binary);
	if (!file.is_open())
	{
		throw std::runtime_error(std::string("failed to open file: ") + absFilepath);
	}

	const size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);
	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}