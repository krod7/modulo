/* ModuloSetup.exe: unpacks the embedded app package to %TEMP% and runs the PowerShell installer. */
#include <windows.h>
#include <stdio.h>
#include "payload.h"

int main(void) {
    char temp[MAX_PATH], dir[MAX_PATH + 32], zip[MAX_PATH + 64], cmd[4 * MAX_PATH];
    GetTempPathA(MAX_PATH, temp);
    snprintf(dir, sizeof dir, "%sModuloSetup", temp);
    CreateDirectoryA(dir, NULL);
    snprintf(zip, sizeof zip, "%s\\payload.zip", dir);

    SetConsoleTitleA("Modulo Setup");
    printf("Preparing Modulo installer...\n");
    FILE *f = fopen(zip, "wb");
    if (!f || fwrite(PAYLOAD, 1, PAYLOAD_LEN, f) != PAYLOAD_LEN) {
        printf("Could not write to %s\n", dir);
        system("pause");
        return 1;
    }
    fclose(f);

    snprintf(cmd, sizeof cmd,
        "powershell -NoProfile -ExecutionPolicy Bypass -Command \""
        "Expand-Archive -LiteralPath '%s' -DestinationPath '%s\\pkg' -Force; "
        "& '%s\\pkg\\installer\\install-windows.ps1'; exit $LASTEXITCODE\"", zip, dir, dir);
    int rc = system(cmd);
    printf("\n");
    system("pause");
    return rc;
}
