#include "ConfigurationManager.h"

ConfigurationManager::ConfigurationManager()
{
}

ConfigurationManager::~ConfigurationManager()
{
	Save();
}

bool ConfigurationManager::CheckOrOpen(const std::string& filename)
{
	if (files.find(filename) == files.end()) // File wasn't loaded yet
	{ 
		// Try to open
		struct stat buf;
		mINI::INIStructure newStructure;
		if (stat(filename.c_str(), &buf) != 0) 
		{
			std::ofstream createFile(filename);
			if (!createFile.is_open())
			{
				throw BadConfigurationFileException(filename);
				return false; // File does not exist and could not be created
			}
		}
		else 
		{
			if (!mINI::INIFile(filename).read(newStructure))
			{
				throw BadConfigurationFileException(filename); // Cannot continue
				return false; // Read failed
			}
		}

		files.emplace(filename, newStructure);
		return true; // Created a new file with empty structure
	}
	// if file exists in memory we're good

	return true;
}

void ConfigurationManager::Save()
{
	for (auto& pair : files) 
	{
		mINI::INIFile file(pair.first);

		file.write(pair.second, true);
	}
}
