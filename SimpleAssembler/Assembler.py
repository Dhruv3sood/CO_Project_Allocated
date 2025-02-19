#!/usr/bin/env python3
import sys


def error_exit(message: str) -> None:
    """Print an error message to stderr and exit."""
    sys.stderr.write(f"ERROR: {message}\n")
    sys.exit(1)


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


def int_to_bin(num: int, bits: int) -> str:
    """
    Convert a signed integer to a two's-complement binary string with 'bits' bits.
    Exits if the number is out of the representable range.
    """
    min_val = -(1 << (bits - 1))
    max_val = (1 << (bits - 1)) - 1
    if num < min_val or num > max_val:
        error_exit(f"Immediate {num} out of range for {bits}-bit signed field.")
    if num < 0:
        num = (1 << bits) + num
    return format(num, f"0{bits}b")

def get_register(reg: str, line: int) -> str:
    """Return the binary string for a given register name."""
    reg = reg.lower().strip()
    if reg not in REGISTERS:
        error_exit(f"Line {line}: Unknown register '{reg}'.")
    return REGISTERS[reg]

def get_immediate(imm: str, line: int) -> int:
    """Convert an immediate string (decimal/hex, maybe negative) into an integer."""
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
        error_exit(f"Line {line}: Invalid immediate '{imm}'.")
    return -value if is_negative else value

from typing import Tuple

def parse_memory_operand(operand: str, line: int) -> Tuple[str, str]:
    """
    Split a memory operand in the form 'imm(rs)' into its components.
    Returns a tuple (imm, rs).
    """
    if "(" not in operand or ")" not in operand:
        error_exit(f"Line {line}: Operand '{operand}' must be in the format imm(rs).")
    imm_part, reg_part = operand.split("(", 1)
    reg_part = reg_part.replace(")", "").strip()
    return imm_part.strip(), reg_part


# Instruction Assembly Functions

def assemble_r_type(mnem: str, operands: list, line: int) -> str:
    """Build the binary encoding for an R-type instruction."""
    if len(operands) != 3:
        error_exit(f"Line {line}: R-type '{mnem}' expects 3 operands.")
    rd = get_register(operands[0], line)
    rs1 = get_register(operands[1], line)
    rs2 = get_register(operands[2], line)
    instr = R_INSTR[mnem]
    return instr["funct7"] + rs2 + rs1 + instr["funct3"] + rd + instr["opcode"]

def assemble_i_type(mnem: str, operands: list, line: int) -> str:
    """Assemble an I-type instruction (lw, addi, jalr)."""
    instr = I_INSTR[mnem]
    opcode, funct3 = instr["opcode"], instr["funct3"]
    if mnem == "lw":
        if len(operands) != 2:
            error_exit(f"Line {line}: 'lw' must have format: rd, imm(rs).")
        rd = get_register(operands[0], line)
        imm_str, rs = parse_memory_operand(operands[1], line)
        rs_bin = get_register(rs, line)
        imm_bin = int_to_bin(get_immediate(imm_str, line), 12)
        return imm_bin + rs_bin + funct3 + rd + opcode
    else:
        if len(operands) != 3:
            error_exit(f"Line {line}: '{mnem}' requires 3 operands: rd, rs, imm.")
        rd = get_register(operands[0], line)
        rs = get_register(operands[1], line)
        imm_bin = int_to_bin(get_immediate(operands[2], line), 12)
        return imm_bin + rs + funct3 + rd + opcode

def assemble_s_type(mnem: str, operands: list, line: int) -> str:
    """Assemble an S-type instruction (sw)."""
    if len(operands) != 2:
        error_exit(f"Line {line}: 'sw' requires 2 operands: rs2, imm(rs).")
    rs2 = get_register(operands[0], line)
    imm_str, rs1 = parse_memory_operand(operands[1], line)
    rs1_bin = get_register(rs1, line)
    imm_bin = int_to_bin(get_immediate(imm_str, line), 12)

    imm_hi = imm_bin[:7]
    imm_lo = imm_bin[7:]
    instr = S_INSTR[mnem]
    return imm_hi + rs2 + rs1_bin + instr["funct3"] + imm_lo + instr["opcode"]

def assemble_b_type(mnem: str, operands: list, line: int, pc: int, labels: dict) -> str:
    """Assemble a B-type instruction (beq, bne, blt)."""
    if len(operands) != 3:
        error_exit(f"Line {line}: B-type '{mnem}' expects 3 operands: rs1, rs2, label/immediate.")
    rs1 = get_register(operands[0], line)
    rs2 = get_register(operands[1], line)
    opcode = B_INSTR[mnem]["opcode"]
    funct3 = B_INSTR[mnem]["funct3"]

    if operands[2] in labels:
        offset = (labels[operands[2]] - pc) >> 1
    else:
        offset = get_immediate(operands[2], line) >> 1

    imm_val = offset << 1

    imm12 = (imm_val >> 12) & 0x1
    imm10_5 = (imm_val >> 5) & 0x3F
    imm4_1 = (imm_val >> 1) & 0xF
    imm11 = (imm_val >> 11) & 0x1
    high_bits = f"{imm12:01b}{imm10_5:06b}"
    low_bits  = f"{imm4_1:04b}{imm11:01b}"
    return high_bits + rs2 + rs1 + funct3 + low_bits + opcode

def assemble_j_type(mnem: str, operands: list, line: int, pc: int, labels: dict) -> str:
    """Assemble a J-type instruction (jal)."""
    if len(operands) != 2:
        error_exit(f"Line {line}: J-type '{mnem}' expects 2 operands: rd, label/immediate.")
    rd = get_register(operands[0], line)
    opcode = J_INSTR[mnem]["opcode"]
    if operands[1] in labels:
        offset = (labels[operands[1]] - pc) >> 1
    else:
        offset = get_immediate(operands[1], line) >> 1
    imm_val = offset << 1

    imm20    = (imm_val >> 20) & 0x1
    imm10_1  = (imm_val >> 1)  & 0x3FF
    imm11    = (imm_val >> 10) & 0x1
    imm19_12 = (imm_val >> 12) & 0xFF
    final_imm = f"{imm20:01b}{imm10_1:010b}{imm11:01b}{imm19_12:08b}"
    return final_imm + rd + opcode

def assemble_instruction(line_str: str, line_num: int, labels: dict, pc: int) -> str:
    """Dispatch the given instruction line to the correct assembly function."""
    tokens = line_str.replace(",", " ").split()
    if not tokens:
        error_exit(f"Line {line_num}: No instruction found.")
    mnem = tokens[0].lower()
    ops = tokens[1:]
    if mnem in R_INSTR:
        return assemble_r_type(mnem, ops, line_num)
    elif mnem in I_INSTR:
        return assemble_i_type(mnem, ops, line_num)
    elif mnem in S_INSTR:
        return assemble_s_type(mnem, ops, line_num)
    elif mnem in B_INSTR:
        return assemble_b_type(mnem, ops, line_num, pc, labels)
    elif mnem in J_INSTR:
        return assemble_j_type(mnem, ops, line_num, pc, labels)
    else:
        error_exit(f"Line {line_num}: Unknown instruction '{mnem}'.")


# Main Program

def main() -> None:
    if len(sys.argv) != 3:
        error_exit("Usage: python Assembler.py <input_file> <output_file>")

    in_fname, out_fname = sys.argv[1], sys.argv[2]
    labels = {}
    instr_list = []
    pc = 0  # Program counter (in bytes)

    # First pass: Read file and capture labels and instructions.
    try:
        with open(in_fname, "r") as infile:
            raw_lines = infile.readlines()
    except FileNotFoundError:
        error_exit(f"Cannot open input file '{in_fname}'")

    for idx, line in enumerate(raw_lines, start=1):
        trimmed = line.strip()
        if not trimmed:
            continue
        # If a colon is present, treat it as a label line.
        if ":" in trimmed:
            lbl, rest = trimmed.split(":", 1)
            lbl = lbl.strip()
            if not lbl or not lbl[0].isalpha():
                error_exit(f"Line {idx}: Invalid label '{lbl}'.")
            if lbl in labels:
                error_exit(f"Line {idx}: Label '{lbl}' is already defined.")
            labels[lbl] = pc
            if rest.strip():
                instr_list.append((idx, pc, rest.strip()))
                pc += 4
        else:
            instr_list.append((idx, pc, trimmed))
            pc += 4

    if not instr_list:
        error_exit("No instructions found.")

    # Second pass: Assemble each instruction.
    assembled_lines = []
    for (ln, addr, instr_text) in instr_list:
        assembled_lines.append(assemble_instruction(instr_text, ln, labels, addr))

    # Check that the final instruction is a virtual halt ("beq zero, zero, 0")
    final_inst = instr_list[-1][2].lower().replace(" ", "").replace(",", "")
    if not (final_inst.startswith("beqzerozero") or final_inst.startswith("beqx0x0")):
        error_exit("Last instruction must be 'beq zero, zero, 0' (virtual halt).")

    # Write the binary output.
    try:
        with open(out_fname, "w") as outfile:
            for code in assembled_lines:
                outfile.write(code + "\n")
    except IOError:
        error_exit(f"Cannot write to output file '{out_fname}'")

    print(f"Assembly complete. Output written to '{out_fname}'.")

if __name__ == "__main__":
    main()
