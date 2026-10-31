// Applies nxrecomp's knowledge to a module imported with the Switch loader:
// function starts (one module-relative hex offset per line) and names
// ("name<TAB>0xoffset" lines, '#' comments). Run as a headless -preScript:
//   analyzeHeadless <proj> <name> -import sdk.bin -preScript ApplySeeds.java <starts.txt> <names.tsv>
//@category Switch
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.symbol.SourceType;
import java.nio.file.Files;
import java.nio.file.Path;

public class ApplySeeds extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        Address base = currentProgram.getImageBase();
        int funcs = 0, failed = 0, names = 0;
        for (String line : Files.readAllLines(Path.of(args[0]))) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            Address a = base.add(Long.parseUnsignedLong(line.replace("0x", ""), 16));
            if (getFunctionAt(a) != null) continue;
            disassemble(a);
            if (createFunction(a, null) != null) funcs++; else failed++;
        }
        if (args.length > 1) {
            for (String line : Files.readAllLines(Path.of(args[1]))) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] f = line.split("\t");
                if (f.length < 2 || f[1].equals("-")) continue;
                Address a = base.add(Long.parseUnsignedLong(f[1].replace("0x", ""), 16));
                createLabel(a, f[0], true, SourceType.USER_DEFINED);
                names++;
            }
        }
        println(String.format("ApplySeeds: %d functions created (%d failed), %d names applied", funcs, failed, names));
    }
}
