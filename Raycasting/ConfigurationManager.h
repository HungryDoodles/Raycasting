#pragma once
#include "Header.h"

#define MINI_CASE_SENSITIVE
#include "mini/ini.h"

#include <string>
#include <map>
#include <sstream>
#include <sys/stat.h>

class BadConfigurationFileException : public std::exception 
{
public:
	BadConfigurationFileException(const std::string& filename) : filename(filename) {}
	
	std::string filename;

	const char* what() const override
	{
		return "Configuration file with the provided filename could not be open or parsed correctly";
	}
};

class ConfigurationManager 
{
public:
	ConfigurationManager();
	virtual ~ConfigurationManager();

	template <typename T>
	T GetOrDefault(const std::string& filename, const std::string& section, const std::string& key, const T& defaultValue);

	template <typename T>
	bool Set(const std::string& filename, const std::string& section, const std::string& key, const T& value);

	bool CheckOrOpen(const std::string& filename);

	void Save();

private:

	std::map<std::string, mINI::INIStructure> files;
};



template<typename T>
inline T ConfigurationManager::GetOrDefault(const std::string& filename, const std::string& section, const std::string& key, const T& defaultValue)
{
	if (!CheckOrOpen(filename)) 
	{
		return defaultValue;
	}

	auto pair = files.find(filename);
	if (!(*pair).second.has(section) || !(*pair).second[section].has(key))
	{
		std::ostringstream ss;
		ss << defaultValue;
		(*pair).second[section][key] = (std::ostringstream() << defaultValue).str();
		return defaultValue;
	}

	T value;
	std::stringstream ss((*pair).second[section][key]);
	if (!(ss >> value))
	{
		(*pair).second[section][key] = (std::ostringstream() << defaultValue).str();
		return defaultValue;
	}
	return value;
}

template<typename T>
inline bool ConfigurationManager::Set(const std::string& filename, const std::string& section, const std::string& key, const T& value)
{
	if (!CheckOrOpen(filename)) 
	{
		return false;
	}

	auto pair = files.find(filename);
	std::stringstream ss;
	bool bSuccess = bool(ss << value);
	(*pair).second[section][key] = ss.str();

	return bSuccess;
}

template<>
inline std::string ConfigurationManager::GetOrDefault(const std::string& filename, const std::string& section, const std::string& key, const std::string& defaultValue)
{
	if (!CheckOrOpen(filename)) 
	{
		return defaultValue;
	}

	auto pair = files.find(filename);

	if (!(*pair).second.has(section) && !(*pair).second[section].has(key)) 
		(*pair).second[section][key] = defaultValue;
	return (*pair).second[section][key];
}

template<>
inline bool ConfigurationManager::Set(const std::string& filename, const std::string& section, const std::string& key, const std::string& value)
{
	if (!CheckOrOpen(filename))
	{
		return false;
	}

	(*files.find(filename)).second[section][key] = value;

	return true;
}