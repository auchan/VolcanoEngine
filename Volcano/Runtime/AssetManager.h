#pragma once
#include <string>
#include <vector>

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

		static std::vector<char> readFile(const std::string& filepath);
	};
}
