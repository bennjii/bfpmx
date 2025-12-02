#include <format>
#include <iostream>
int main() {
    std::string name = "World";
    std::string message = std::format("Hello, {}!", name);
    std::cout << message << std::endl;
    return 0;
}