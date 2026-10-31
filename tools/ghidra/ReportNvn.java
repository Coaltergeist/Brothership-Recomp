// Prints a summary of an analysed module and decompiles the named functions.
//   analyzeHeadless <proj> <name> -process sdk.bin -noanalysis -readOnly -postScript ReportNvn.java <name>...
//@category Switch
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.Symbol;

public class ReportNvn extends GhidraScript {
    @Override
    public void run() throws Exception {
        println("functions: " + currentProgram.getFunctionManager().getFunctionCount());
        DecompInterface d = new DecompInterface();
        d.openProgram(currentProgram);
        for (String name : getScriptArgs()) {
            for (Symbol s : currentProgram.getSymbolTable().getSymbols(name)) {
                Function f = getFunctionAt(s.getAddress());
                println("== " + name + " @ " + s.getAddress() + (f == null ? " (no function)" : ""));
                if (f != null) {
                    println(d.decompileFunction(f, 60, monitor).getDecompiledFunction().getC());
                }
            }
        }
        d.dispose();
    }
}
