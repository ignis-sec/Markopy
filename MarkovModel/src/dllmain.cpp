/** @file dllmain.cpp
 * @brief DLLMain for dynamic windows library
 * @authors Ata Hakçıl
 * 
 * @copydoc Markov::Model
 */

#include "pch.h"
#include "model.h"
#include <iostream>


#ifdef _WIN32
__declspec(dllexport) void dll_loadtest() {
    std::cout << "External function called.\n";
    //cudaTestEntry();
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved)
{
    switch (ul_reason_for_call)
    {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

#endif

