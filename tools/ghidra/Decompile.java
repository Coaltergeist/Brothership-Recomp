// Decompiles functions of an analysed program, by name or address, with their
// callers and callees. Read-only.
//   analyzeHeadless <proj> <name> -process sdk.bin -noanalysis -readOnly \
//       -scriptPath tools/ghidra -postScript Decompile.java <name|0xaddr>...
// An address may be inside a function; the containing function is shown.
//@category Switch
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.Symbol;
import java.util.ArrayList;
import java.util.List;

public class Decompile extends GhidraScript {
    @Override
    public void run() throws Exception {
        DecompInterface d = new DecompInterface();
        d.openProgram(currentProgram);
        for (String arg : getScriptArgs()) {
            List<Function> fs = new ArrayList<>();
            if (arg.startsWith("0x")) {
                Address a = toAddr(Long.parseUnsignedLong(arg.substring(2), 16));
                Function f = getFunctionContaining(a);
                if (f != null) fs.add(f);
            } else {
                for (Symbol s : currentProgram.getSymbolTable().getSymbols(arg)) {
                    Function f = getFunctionAt(s.getAddress());
                    if (f != null) fs.add(f);
                }
                if (fs.isEmpty()) {
                    // Demangled C++ names: match the function name anywhere in the listing.
                    for (Function f : currentProgram.getFunctionManager().getFunctions(true)) {
                        if (f.getName(true).equals(arg) || f.getName().equals(arg)) fs.add(f);
                    }
                }
            }
            if (fs.isEmpty()) {
                println("== " + arg + ": not found");
                continue;
            }
            for (Function f : fs) {
                StringBuilder sb = new StringBuilder();
                sb.append("== ").append(f.getName(true)).append(" @ ").append(f.getEntryPoint()).append("\n");
                sb.append("callers:");
                for (Function c : f.getCallingFunctions(monitor)) sb.append(" ").append(c.getName(true)).append("@").append(c.getEntryPoint());
                sb.append("\ncallees:");
                for (Function c : f.getCalledFunctions(monitor)) sb.append(" ").append(c.getName(true)).append("@").append(c.getEntryPoint());
                sb.append("\n").append(d.decompileFunction(f, 120, monitor).getDecompiledFunction().getC());
                println(sb.toString());
            }
        }
        d.dispose();
    }
}
