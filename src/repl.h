#ifndef _REPL_COMMAND_CONSOLE__
#define _REPL_COMMAND_CONSOLE__

#include <unordered_map>

class ReplCommandConsole {
public:
    
    typedef void (*CommandFunc)(const std::vector<std::string>&);
    
    bool Run(const std::string& command, const std::vector<std::string>& arguments) {
        std::unordered_map<std::string, CommandFunc>::iterator it = mCommand.find(command);
        if (it == mCommand.end()) {
            return false;
        }
        
        CommandFunc func = it->second;
        if (func) {
            func(arguments);
        }
        return true;
    }
    
    bool RegisterCommandFunction(const std::string& name, CommandFunc commandPtr) {
        if (commandPtr == NULL) 
            return false;
        
        if (mCommand.find(name) != mCommand.end()) 
            return false;
        
        mCommand[name] = commandPtr;
        return true;
    }
    
private:
    
    std::unordered_map<std::string, CommandFunc> mCommand;
};

#endif
