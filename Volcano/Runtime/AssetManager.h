#pragma once
#include <string>

namespace volcano
{
	class AssetManager
	{
	public:
		static AssetManager instance()
		{
			static AssetManager instanceObj;
			return instanceObj;
		}
		
		static std::string getAssetPath(const std::string &relativePath);
	};
}
