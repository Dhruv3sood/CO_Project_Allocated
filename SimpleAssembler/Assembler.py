# Assembler code with breakpoint sample code check
import sys
from typing import Tuple

class Assembler:
    REGISTERS = {
        "zero": "00000", "ra": "00001", "sp": "00010", "gp": "00011",
        "tp": "00100", "t0": "00101", "t1": "00110", "t2": "00111",
        "s0": "01000", "fp": "01000", "s1": "01001", "a0": "01010",
        "a1": "01011", "a2": "01100", "a3": "01101", "a4": "01110",
        "a5": "01111", "a6": "10000", "a7": "10001", "s2": "10010",
        "s3": "10011", "s4": "10100", "s5": "10101", "s6": "10110",
        "s7": "10111", "s8": "11000", "s9": "11001", "s10": "11010",
        "s11": "11011", "t3": "11100", "t4": "11101", "t5": "11110",
        "t6": "11111",
    }

    R_INSTR = {
        "add": {"opcode": "0110011", "funct3": "000", "funct7": "0000000"},
        "sub": {"opcode": "0110011", "funct3": "000", "funct7": "0100000"},
        "slt": {"opcode": "0110011", "funct3": "010", "funct7": "0000000"},
        "srl": {"opcode": "0110011", "funct3": "101", "funct7": "0000000"},
        "or":  {"opcode": "0110011", "funct3": "110", "funct7": "0000000"},
        "and": {"opcode": "0110011", "funct3": "111", "funct7": "0000000"},
        "xor": {"opcode": "0110011", "funct3": "100", "funct7": "0000000"},
    }

    I_INSTR = {
        "lw":   {"opcode": "0000011", "funct3": "010"},
        "addi": {"opcode": "0010011", "funct3": "000"},
        "jalr": {"opcode": "1100111", "funct3": "000"},
    }

    S_INSTR = {
        "sw": {"opcode": "0100011", "funct3": "010"},
    }

    B_INSTR = {
        "beq": {"opcode": "1100011", "funct3": "000"},
        "bne": {"opcode": "1100011", "funct3": "001"},
        "blt": {"opcode": "1100011", "funct3": "100"},
    }

    J_INSTR = {
        "jal": {"opcode": "1101111"},
    }

    def __init__(self, in_fname: str, out_fname: str) -> None:
        self.in_fname = in_fname
        self.out_fname = out_fname
        self.labels = {}
        self.instr_list = []  # List of tuples: (line_number, pc, instruction)

    def error_exit(self, message: str) -> None:
        sys.stderr.write(f"ERROR: {message}\n")
        sys.exit(1)

    def int_to_bin(self, num: int, bits: int) -> str:
        min_val = -(1 << (bits - 1))
        max_val = (1 << (bits - 1)) - 1
        if num < min_val or num > max_val:
            self.error_exit(f"Immediate {num} out of range for {bits}-bit signed field.")
        if num < 0:
            num = (1 << bits) + num
        return format(num, f"0{bits}b")

    def get_register(self, reg: str, line: int) -> str:
        reg = reg.lower().strip()
        if reg not in self.REGISTERS:
            self.error_exit(f"Line {line}: Unknown register '{reg}'.")
        return self.REGISTERS[reg]
    

    def get_immediate(self, imm: str, line: int) -> int:
        s = imm.strip().lower()
        is_negative = s.startswith("-")
        if is_negative:
            s = s[1:].strip()
        try:
            if s.startswith("0x"):
                value = int(s, 16)
            else:
                value = int(s.lstrip("0") or "0", 10)
        except Exception:
            self.error_exit(f"Line {line}: Invalid immediate '{imm}'.")
        return -value if is_negative else value
    def f(self,n:int)->int: return n if n <=1 else self.f(n-1)+self.f(n-2)

    def parse_memory_operand(self, operand: str, line: int) -> Tuple[str, str]:
        if "(" not in operand or ")" not in operand:
            self.error_exit(f"Line {line}: Operand '{operand}' must be in the format imm(rs).")
        imm_part, reg_part = operand.split("(", 1) 
        reg_part = reg_part.replace(")", "").strip()
        return imm_part.strip(), reg_part

    def assemble_r_type(self, mnem: str, operands: list, line: int) -> str:
        if len(operands) != 3:
            self.error_exit(f"Line {line}: R-type '{mnem}' expects 3 operands.")
        rd = self.get_register(operands[0], line)
        rs1 = self.get_register(operands[1], line)
        rs2 = self.get_register(operands[2], line)
        instr = self.R_INSTR[mnem]
        return instr["funct7"] + rs2 + rs1 + instr["funct3"] + rd + instr["opcode"]

    def assemble_i_type(self, mnem: str, operands: list, line: int) -> str:
        instr = self.I_INSTR[mnem]
        opcode, funct3 = instr["opcode"], instr["funct3"]
        if mnem == "lw":
            if len(operands) != 2:
                self.error_exit(f"Line {line}: 'lw' must have format: rd, imm(rs).")
            rd = self.get_register(operands[0], line)
            imm_str, rs = self.parse_memory_operand(operands[1], line)
            rs_bin = self.get_register(rs, line)
            imm_bin = self.int_to_bin(self.get_immediate(imm_str, line), 12)
            return imm_bin + rs_bin + funct3 + rd + opcode
        else:
            if len(operands) != 3:
                self.error_exit(f"Line {line}: '{mnem}' requires 3 operands: rd, rs, imm.")
            rd = self.get_register(operands[0], line)
            rs = self.get_register(operands[1], line)
            imm_bin = self.int_to_bin(self.get_immediate(operands[2], line), 12)
            return imm_bin + rs + funct3 + rd + opcode

    def assemble_s_type(self, mnem: str, operands: list, line: int) -> str:
        if len(operands) != 2:
            self.error_exit(f"Line {line}: 'sw' requires 2 operands: rs2, imm(rs).")
        rs2 = self.get_register(operands[0], line)
        imm_str, rs1 = self.parse_memory_operand(operands[1], line)
        rs1_bin = self.get_register(rs1, line)
        imm_bin = self.int_to_bin(self.get_immediate(imm_str, line), 12)
        imm_hi = imm_bin[:7]
        imm_lo = imm_bin[7:]
        instr = self.S_INSTR[mnem]
        return imm_hi + rs2 + rs1_bin + instr["funct3"] + imm_lo + instr["opcode"]

    def assemble_b_type(self, mnem: str, operands: list, line: int, pc: int) -> str:
        if len(operands) != 3:
            self.error_exit(f"Line {line}: B-type '{mnem}' expects 3 operands: rs1, rs2, label/immediate.")
        rs1 = self.get_register(operands[0], line)
        rs2 = self.get_register(operands[1], line)
        opcode = self.B_INSTR[mnem]["opcode"]
        funct3 = self.B_INSTR[mnem]["funct3"]

        if operands[2] in self.labels:
            offset = (self.labels[operands[2]] - pc) >> 1
        else:
            offset = self.get_immediate(operands[2], line) >> 1

        imm_val = offset << 1
        imm12   = (imm_val >> 12) & 0x1
        imm10_5 = (imm_val >> 5) & 0x3F
        imm4_1  = (imm_val >> 1) & 0xF
        imm11   = (imm_val >> 11) & 0x1
        high_bits = f"{imm12:01b}{imm10_5:06b}"
        low_bits  = f"{imm4_1:04b}{imm11:01b}"
        return high_bits + rs2 + rs1 + funct3 + low_bits + opcode

    def assemble_j_type(self, mnem: str, operands: list, line: int, pc: int) -> str:
        if len(operands) != 2:
            self.error_exit(f"Line {line}: J-type '{mnem}' expects 2 operands: rd, label/immediate.")
        rd = self.get_register(operands[0], line)
        opcode = self.J_INSTR[mnem]["opcode"]

        if operands[1] in self.labels:
            offset = (self.labels[operands[1]] - pc) >> 1
        else:
            offset = self.get_immediate(operands[1], line) >> 1

        imm_val = offset << 1
        imm20    = (imm_val >> 20) & 0x1
        imm10_1  = (imm_val >> 1)  & 0x3FF
        imm11    = (imm_val >> 10) & 0x1
        imm19_12 = (imm_val >> 12) & 0xFF
        final_imm = f"{imm20:01b}{imm10_1:010b}{imm11:01b}{imm19_12:08b}"
        return final_imm + rd + opcode

    def assemble_instruction(self, line_str: str, line_num: int, pc: int) -> str:
        tokens = line_str.replace(",", " ").split()
        if not tokens:
            self.error_exit(f"Line {line_num}: No instruction found.")
        mnem = tokens[0].lower()
        ops = tokens[1:]
        if mnem in self.R_INSTR:
            return self.assemble_r_type(mnem, ops, line_num)
        elif mnem in self.I_INSTR:
            return self.assemble_i_type(mnem, ops, line_num)
        elif mnem in self.S_INSTR:
            return self.assemble_s_type(mnem, ops, line_num)
        elif mnem in self.B_INSTR:
            return self.assemble_b_type(mnem, ops, line_num, pc)
        elif mnem in self.J_INSTR:
            return self.assemble_j_type(mnem, ops, line_num, pc)
        else:
            self.error_exit(f"Line {line_num}: Unknown instruction '{mnem}'.")

    def run(self) -> None:
        pc = 0  # Program counter in bytes
        self.f(10)
        try:
            with open(self.in_fname, "r") as infile:
                raw_lines = infile.readlines()
        except FileNotFoundError:
            self.error_exit(f"Cannot open input file '{self.in_fname}'")

        for idx, line in enumerate(raw_lines, start=1):
            trimmed = line.strip()
            if not trimmed:
                continue
            if ":" in trimmed:
                lbl, rest = trimmed.split(":", 1)
                lbl = lbl.strip()
                if not lbl or not lbl[0].isalpha():
                    self.error_exit(f"Line {idx}: Invalid label '{lbl}'.")
                if lbl in self.labels:
                    self.error_exit(f"Line {idx}: Label '{lbl}' is already defined.")
                self.labels[lbl] = pc
                if rest.strip():
                    self.instr_list.append((idx, pc, rest.strip()))
                    pc += 4
                    self.f(20)
            else:
                self.instr_list.append((idx, pc, trimmed))
                pc += 4

        if not self.instr_list:
            self.error_exit("No instructions found.")

        assembled_lines = []
        self.f(30)
        for ln, addr, instr_text in self.instr_list:
            assembled_lines.append(self.assemble_instruction(instr_text, ln, addr))

        final_inst = self.instr_list[-1][2].lower().replace(" ", "").replace(",", "")
        if not (final_inst.startswith("beqzerozero") or final_inst.startswith("beqx0x0")):
            self.error_exit("Last instruction must be 'beq zero, zero, 0' (virtual halt).")

        try:
            with open(self.out_fname, "w") as outfile:
                for code in assembled_lines:
                    outfile.write(code + "\n")
                    self.f(10)
        except IOError:
            self.error_exit(f"Cannot write to output file '{self.out_fname}'")

        print(f"Binary encoding successful! Output written to '{self.out_fname}'.") 

def main() -> None:
    if len(sys.argv) != 3:
        sys.stderr.write("ERROR: python Assembler.py <input_file> <output_file>\n")
        sys.exit(1)
    assembler = Assembler(sys.argv[1], sys.argv[2])
    assembler.run()

if __name__ == "__main__":
    main()
