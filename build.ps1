rm -r build
mkdir build

cmake -G "Visual Studio 17 2022" -A x64 -S . -B build
cmake --build build --config Release
