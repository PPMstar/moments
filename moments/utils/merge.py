"""
The goal of this script is to replace a subroutine of one version of the PPM code
with the matching subroutine of another version
"""

import warnings
import numpy
import sys
import os
import re
import moments.core.wrapper
from . import fpp

__all__ = ["merge_code", "update_domain"]

def merge_code(target, source, subroutine, output=None):
    target = source_code(target)
    source = source_code(source)

    msg_merge_start = "\nc Automatically merged from File: \"" + os.path.basename(source.fname) + "\"."
    msg_merge_start = msg_merge_start + ("-" * max(0, 72 - len(msg_merge_start))) + "\n"
    msg_merge_end = "\nc End of merged code."
    msg_merge_end = msg_merge_end + ("-" * max(0, 72 - len(msg_merge_end))) + "\n"

    source_call_context = source.get_call_context(subroutine)
    target_call_context = target.get_call_context(subroutine, get_linenum=True)

    temp_subroutines = [subroutine]
    subroutines = []
    subroutines_source = "\n\n\n"
    subroutine_definitions = set()
    while len(temp_subroutines) > 0:
        routine_source, calls_source, definitions = source.get_subroutine(temp_subroutines[0])
        subroutine_definitions = subroutine_definitions | definitions
        subroutines_source = subroutines_source + routine_source + "\n\n\n"
        subroutines.append(temp_subroutines[0])
        temp_subroutines = temp_subroutines[1:] + [routine for routine in calls_source if (routine not in subroutines) and (routine not in temp_subroutines[1:])] 

    routine_target = target.remove_subroutine(subroutines, target_call_context, source_call_context, msg_merge_start, msg_merge_end)

    missing_definitions = subroutine_definitions - (subroutine_definitions & set(target.definitions.keys()))
    definitions_source = source.get_definitions(missing_definitions) + "\n\n\n"

    if output is None:
        output = "./" + os.path.basename(target.fname) + ".automerge"
        if not os.path.isfile(output):
            print("Saving merged version as \"" + output + "\"")

    if os.path.isfile(output):
        while True:
            sys.stdout.write("File: \"" + output + "\" already exists. Would you like to overwrite? [Y/n]: ")
            choice = input().lower().strip()
            if choice in "yes":
                break
            elif choice in "no":
                return

    definitions_source = msg_merge_start + definitions_source + msg_merge_end
    subroutines_source = msg_merge_start + subroutines_source + msg_merge_end
    with open(output, "w") as fout:
        fout.write(definitions_source + routine_target + subroutines_source)


def update_domain(source, target, code):
    read_ppm_in = ""
    with open(source, "r") as fin:
        read_ppm_in = fin.read()

    definitions = fpp.preprocess(code)

    if ("nnzteams" in definitions) and ("nnxteams" not in definitions):
        definitions["nnxteams"] = definitions["nnzteams"]

    if ("nntzbricks" in definitions) and ("nntxbricks" not in definitions):
        definitions["nntxbricks"] = definitions["nntzbricks"]

    if ("nnnnz" in definitions) and ("nnnnx" not in definitions):
        definitions["nnnnx"] = definitions["nnnnz"]

    nnnnx, nnnny, nnnnz = definitions["nnnnx"], definitions["nnnny"], definitions["nnnnz"]
    nntxbricks, nntybricks, nntzbricks = definitions["nntxbricks"], definitions["nntybricks"], definitions["nntzbricks"]
    nnxteams, nnyteams, nnzteams = definitions["nnxteams"], definitions["nnyteams"], definitions["nnzteams"]

    rradmax = definitions["rradmax"]

    update_read_ppm = {}

    resolutionx = int(nnnnx*nnxteams*nntxbricks/4.0)
    resolutiony = int(nnnny*nnyteams*nntybricks/4.0)
    resolutionz = int(nnnnz*nnzteams*nntzbricks/4.0)

    update_read_ppm["ixyz0"] = "  1   1   1"
    update_read_ppm["nxyz"] = str(resolutionx) + " " + str(resolutiony) + " " + str(resolutionz)
    update_read_ppm["nblend"] = 1

    update_read_ppm["bsizex"] = 2.0*rradmax/float(nntxbricks*nnxteams)
    update_read_ppm["bsizey"] = 2.0*rradmax/float(nntybricks*nnyteams)
    update_read_ppm["bsizez"] = 2.0*rradmax/float(nntzbricks*nnzteams)

    update_read_ppm["nbrkx"] = nntxbricks
    update_read_ppm["nbrky"] = nntybricks
    update_read_ppm["nbrkz"] = nntzbricks

    if (nntxbricks % 2) == 0:
        update_read_ppm["nodsx"] = int(nntxbricks/2)
    else:
        update_read_ppm["nodsx"] = nntxbricks

    if (nntybricks % 2) == 0:
        update_read_ppm["nodsy"] = int(nntybricks/2)
    else:
        update_read_ppm["nodsy"] = nntybricks

    if (nntzbricks % 2) == 0:
        update_read_ppm["nodsz"] = int(nntzbricks/2)
    else:
        update_read_ppm["nodsz"] = nntzbricks

    update_read_ppm["ndbx"] = 1
    update_read_ppm["ndby"] = nntxbricks
    update_read_ppm["ndbz"] = nntxbricks*nntybricks

    update_read_ppm["nfilex"] = nnxteams
    update_read_ppm["nfiley"] = nnyteams
    update_read_ppm["nfilez"] = nnzteams

    update_read_ppm["ncpbx"] = nnnnx/4
    update_read_ppm["ncpby"] = nnnny/4
    update_read_ppm["ncpbz"] = nnnnz/4

    update_read_ppm["nbpb"] = 2

    read_ppm_in = moments.core.wrapper._update_read_ppm_in(read_ppm_in, **update_read_ppm)

    with open(target, "w") as fout:
        fout.write(read_ppm_in)

class source_code:
    _f77_comments = ["c", "C", "*", "!"]
    _string = ["'", '"']
    _fpp_if = ["#if", "#ifdef", "#ifndef"]
    _call_context_width = 16

    def __init__(self, fname, include_fpp=True):
        self.fname = fname
        self.include_fpp = include_fpp

        self.source = None
        self.subroutine_tokens = {} # format "token":(start, end, set("calls"), set("definitions"))
        self.subroutine_call_context = {} # format "token":[(start, call_start, call_end, end)]
        self.definitions = {}

        self._get_source()
        self._get_definitions()
        self._find_subroutines()
        self._check_preprocessor()
        self._find_calls()

    def __str__(self):
        return "\n".join(self.source)

    def get_definitions(self, definitions):
        return "\n".join([self.source[self.definitions[definition]] for definition in sorted(definitions)])

    def get_subroutine(self, token):
        if token in self.subroutine_tokens:
            start, end, calls, definitions = self.subroutine_tokens[token]
            calls = [subroutine for subroutine in calls if subroutine in self.subroutine_tokens]
            return "\n".join(self.source[start:end]), calls, definitions
        else:
            raise NameError("File: \"" + self.fname + "\", subroutine not found \"" + token + "\"")

    def get_call_context(self, token, get_linenum=False):
        call_context = [] # [(subroutine, precall, call, postcall)]

        for subroutine in self.subroutine_call_context:
            for context_start, call_start, call_end, context_end in self.subroutine_call_context[subroutine]:
                precall_code, call_code, postcall_code = self.source[context_start:call_start], self.source[call_start:call_end], self.source[call_end:context_end]

                found_call = False
                for i, line in enumerate(call_code):
                    line_tokens = self._remove_comments([line])
                    if len(line_tokens) == 0:
                        line_tokens = ""
                    else:
                        line_tokens = line_tokens[0]
                    line_tokens = re.findall(r"[#\w]+", line_tokens.strip())        
                    if len(line_tokens) > 1:
                        if line_tokens[0] == "call":
                            if line_tokens[1] == token:
                                found_call = True
                                break
                if found_call:
                    if get_linenum:
                        call_context.append((subroutine, context_start, call_start, call_end, context_end))
                    else:
                        call_context.append((subroutine, precall_code, call_code, postcall_code))
        return call_context

    def remove_subroutine(self, tokens, target_call_context, source_call_context, msg_merge_start="", msg_merge_end=""):
        source_tokens = [] # [(start, stop, new_code)]

        for target_subroutine, context_start, call_start, call_end, context_end in target_call_context:
            found_match = False
            for source_subroutine, source_precall, source_call, source_postcall in source_call_context:
                precall_code, call_code, postcall_code = self.source[context_start:call_start], self.source[call_start:call_end], self.source[call_end:context_end]
                if (precall_code == source_precall) and (postcall_code == source_postcall):
                    source_tokens.append((call_start, call_end, [msg_merge_start] + source_call + [msg_merge_end]))
                    found_match = True
                    break
            if not found_match:
                self._warning("Target subroutine call(s) do not match call(s) in the source version", call_start, call_end)

        if isinstance(tokens, str): # is list of strings
            tokens = [tokens]

        tokens = [self.subroutine_tokens[token] for token in tokens if token in self.subroutine_tokens]
        tokens = [(start, stop, []) for start, stop, calls, definitions in tokens]
        tokens = tokens + source_tokens

        tokens = sorted(tokens, key=lambda subroutine: subroutine[0])

        lines = []
        if len(tokens) > 0:
            lines = self.source[0:tokens[0][0]]
            if len(tokens[0][2]) > 0:
                lines = lines + tokens[0][2]

            if len(tokens) > 1:
                for i in range(len(tokens) - 1):
                    lines = lines + self.source[tokens[i][1]:tokens[i + 1][0]]
                    if len(tokens[i + 1][2]) > 0:
                        lines = lines + tokens[i + 1][2]

            lines = lines + self.source[tokens[-1][1]:-1]
        return "\n".join(lines)

    def _get_source(self):
        source = ""
        with open(self.fname, "r") as fin:
            source = fin.read()
        self.source = source.split("\n")

    def _find_calls(self):
        calls = []#[(token, line)]
        for i, line in enumerate(self.source):
            line_tokens = self._remove_comments([line])
            if len(line_tokens) == 0:
                line_tokens = ""
            else:
                line_tokens = line_tokens[0]
            line_tokens = re.findall(r"[#\w]+", line_tokens.strip())        
            if len(line_tokens) > 1:
                if line_tokens[0] == "call":
                    calls.append((line_tokens[1], i))

        call_context = [] #[(context start, call start, call end, context end)]

        for call, call_line in calls:
            context_start, call_end, context_end = 0, 0, 0

            precall_code, call_code, postcall_code = [], [], []

            i = max(0, call_line - self._call_context_width)
            precall_code = self.source[i:call_line]
            context_start = i

            i = call_line
            iscall = True

            line = self._remove_comments([self.source[i]])
            if len(line) == 0:
                line = ""
            else:
                line = line[0].strip()

            while iscall:
                call_code.append(self.source[i])

                i += 1
                next_line = self._remove_comments([self.source[i]])
                if len(next_line) == 0:
                    next_line = ""
                else:
                    next_line = next_line[0].strip()

                if len(next_line) > 0:
                    fortran_call = "call " + call
                    if len(next_line) >= len(fortran_call):
                        if (not next_line[0] in ["#", "&"]) and (next_line[:len(fortran_call)] != fortran_call):
                            iscall = False
                    else:
                        if not next_line[0] in ["#", "&"]:
                            iscall = False
            call_end = i

            i = min(len(self.source), call_end + self._call_context_width)
            postcall_code = self.source[call_end:i]
            context_end = i

            if len(call_context) > 0:
                old_context_start, old_call_start, old_call_end, old_context_end = call_context[-1]
                if call_line < old_call_end:
                    continue

            call_context.append((context_start, call_line, call_end, context_end))

        for context_start, call_start, call_end, context_end in call_context:
            for subroutine in self.subroutine_tokens:
                start, end, calls, definitions = self.subroutine_tokens[subroutine]
                if (call_start >= start) and (call_start < end):
                    if not subroutine in self.subroutine_call_context:
                        self.subroutine_call_context[subroutine] = []
                    self.subroutine_call_context[subroutine].append((context_start, call_start, call_end, context_end))

    def _find_subroutines(self):
        token = None
        fppstart = None
        start = None
        end = None
        calls = set()
        definitions = set()
        for i, line in enumerate(self.source):
            line_tokens = self._remove_comments([line])
            if len(line_tokens) == 0:
                line_tokens = ""
            else:
                line_tokens = line_tokens[0]
            line_tokens = re.findall(r"[#\w]+", line_tokens.strip())

            if len(line_tokens) > 0:
                if self.include_fpp:
                    if start is None:
                        if line_tokens[0] == "subroutine":
                            if len(line_tokens) > 1:
                                token, start = line_tokens[1], i
                            else:
                                raise ValueError("File: \"" + self.fname + "\", at Line: " + str(i) + ", found end of statment when expecting subroutine token.\n" + str(i) + ": " + line)
                    else:
                        if line_tokens[0] == "end":
                            end = i + 1
                            if fppstart is not None:
                                start = min(start, fppstart)
                            self.subroutine_tokens[token] = (start, end, calls, definitions)
                            token, fppstart, start, end, calls, definitions =None, None, None, None, set(), set()
                        elif line_tokens[0] == "call":
                            if len(line_tokens) > 1:
                                calls.add(line_tokens[1])
                            else:
                                raise ValueError("File: \"" + self.fname + "\", at Line: " + str(i) + ", found end of statment when expecting subroutine call token.\n" +  str(i) + ": " + line)

                    if (fppstart is None) and (start is None):
                        if line[0] == "#":
                            fppstart = i
                    elif (start is None):
                        if line[0] != "#":
                            fppstart = None
                            definitions = set()

                    if fppstart is not None:
                        for line_token in line_tokens:
                            if line_token in self.definitions:
                                definitions.add(line_token)
                else:
                    if start is None:
                        if line_tokens[0] == "subroutine":
                            if len(line_tokens) > 1:
                                token, start = line_tokens[1], i
                            else:
                                raise ValueError("File: \"" + self.fname + "\", at Line: " + str(i) + ", found end of statment when expecting subroutine token.\n" + str(i) + ": " + line)
                    else:
                        if line_tokens[0] == "end":
                            end = i + 1
                            self.subroutine_tokens[token] = (start, end, calls)
                            token, start, end, calls = None, None, None, set()
                        elif line_tokens[0] == "call":
                            if len(line_tokens) > 1:
                                calls.add(line_tokens[1])
                            else:
                                raise ValueError("File: \"" + self.fname + "\", at Line: " + str(i) + ", found end of statment when expecting subroutine call token.\n" +  str(i) + ": " + line)
            else:
                if self.include_fpp:
                    if start is None:
                        fppstart = None

        if end is not None:
            raise EOFError("File: \"" + self.fname + "\", Unexpected end of file in subroutine \"" + token + "\"")

    def _get_definitions(self):
        if self.include_fpp:
            open_if_blocks = 0
            for i, line in enumerate(self.source):
                line = self._remove_comments([line])
                if len(line) == 0:
                    line = ""
                else:
                    line = line[0]
                line = re.findall(r"[#\w]+", line.strip())

                if len(line) > 0:
                    if line[0] == "#define":
                        self.definitions[line[1]] = i

    def _check_preprocessor(self):
        open_if_blocks = 0
        for i, line in enumerate(self.source):
            line = self._remove_comments([line])
            if len(line) == 0:
                line = ""
            else:
                line = line[0]
            line = re.findall(r"[#\w]+", line.strip())

            if len(line) > 0:
                if line[0] in self._fpp_if:
                    open_if_blocks += 1
                elif line[0] == "#endif":
                    open_if_blocks -= 1

                if open_if_blocks < 0:
                    raise ValueError("File: \"" + self.fname + "\", at Line: " + str(i) + ", #endif without #if.")
        if open_if_blocks > 0:
            raise EOFError("File: \"" + self.fname + "\", #if without #endif.")

    def _remove_comments(self, code):
        code = self._remove_f77_comment(code)
        return self._remove_f90_comment(code)

    def _remove_f77_comment(self, code):
        source = []
        for line in code:
            if len(line) > 0:
                if not (line[0] in self._f77_comments):
                    source.append(line)
        return source

    def _remove_f90_comment(self, code):
        source = []
        for line in code:
            if len(line) > 0:
                indices = [i for i, char in enumerate(line) if char == "!"]
                for index in indices:
                    if not self._in_string(line, index):
                        line = line[:index]
                        break
                source.append(line)
        return source

    def _in_string(self, line, n):
        allblank = True
        is_string = False
        quote_type = None
        for i, char in enumerate(line):
            if n == i:
                break

            if not is_string:
                if allblank:
                    if char in ["!"]:
                        break
                else:
                    if char in ["!", "&"]:
                        break

            if char in self._string:
                if not is_string:
                    is_string = True
                    quote_type = char
                else:
                    if quote_type == char:
                        is_string = False

            if len(char.strip()) == 1:
                allblank = False
        return is_string

    def _warning(self, message, snipet_start, snipet_end, category=UserWarning, buffer=4):
        lines = self._remove_comments(self.source[snipet_start:snipet_end])
        if len(lines) > 2*buffer + 1:
            lines = lines[:buffer] + ["..."] + lines[-buffer:]
        warnings.showwarning(message, category, self.fname, snipet_start, line="\n  ".join(lines))
