#include "AssetManager.h"

using namespace volcano;

std::string AssetManager::getAssetPath(const std::string &relativePath)
{
	return std::string("..\\assets\\") + relativePath;
}