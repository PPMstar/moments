import re

def define(code, **kwargs):
    code = code.split("\n")
    for token, value in kwargs.items():
        not_found = True
        for i, line in enumerate(code):
            if line.startswith("#define"):
                line = line.split()
                if len(line) < 2:
                    raise ValueError("Missing identifier.\n\t\"" + str(code[i]) + "\"")
            
                if token == line[1]:
                    code[i] = "#define " + token + " " + str(value)
                    not_found = False
        
        if not_found:
            code = ["#define " + token + " " + str(value)] + code
    return "\n".join(code)

def preprocess(fname):
    file_contents = ""
    with open(fname, "r") as fin:
        file_contents = fin.read()

    definitions = {}
    
    file_contents = file_contents.split("\n")
    code = []
    for line in file_contents:
        if line.startswith("#"):
            code.append(line)
    
    conditionals=[True]
    for i, line in enumerate(code):
        if line.startswith("#define"):
            line = code[i].split(None, 2)
            if len(line) < 2:
                raise ValueError("Missing identifier.\n\t\"" + str(code[i]) + "\"")
            if len(line) < 3:
                raise ValueError("Missing value.\n\t\"" + str(code[i]) + "\"")
            if "(" in line[1]:
                #TODO: log warning
                print("Warning: Evaluation of preprocessor macros not implemented.\n\t\"" + str(code[i]) + "\"")
                continue

            if conditionals[-1]:
                try:
                    definitions[line[1]] = _evaluate_statement(line[2], definitions)
                except NameError:
                    #TODO: log warning
                    print("Warning: Evaluation of preprocessor directive failed.\n\t\"" + str(code[i]) + "\"")
        elif line.startswith("#undef"):
            line = code[i].split()
            if len(line) < 2:
                raise ValueError("Missing identifier.\n\t\"" + str(code[i]) + "\"")
            
            if conditionals[-1]:
                if line[1] in definitions:
                    del definitions[line[1]]
        elif line.startswith("#ifdef"):
            line = code[i].split()
            if len(line) < 2:
                raise ValueError("Missing identifier.\n\t\"" + str(code[i]) + "\"")
            
            if conditionals[-1]:
                conditionals.append(line[1] in definitions)
            else:
                #is false or None
                conditionals.append(None)
        elif line.startswith("#ifndef"):
            line = code[i].split()
            if len(line) < 2:
                raise ValueError("Missing identifier.\n\t\"" + str(code[i]) + "\"")
            
            if conditionals[-1]:
                conditionals.append(line[1] not in definitions)
            else:
                #is false or None
                conditionals.append(None)
        elif line.startswith("#if"):
            line = code[i].split(None, 1)
            if len(line) < 2:
                raise ValueError("Missing identifier.\n\t\"" + str(code[i]) + "\"")
            
            if conditionals[-1]:
                try:
                    conditionals.append(bool(_evaluate_statement(line[1], definitions)))
                except NameError:
                    #TODO: log warning
                    print("Warning: Evaluation of preprocessor directive failed.\n\t\"" + str(code[i]) + "\"")
                    conditionals.append(False)
            else:
                #is false or None
                conditionals.append(None)
        elif line.startswith("#elif"):
            line = code[i].split(None, 1)
            if len(line) < 2:
                raise ValueError("Missing identifier.\n\t\"" + str(code[i]) + "\"")
            
            last_conditional = conditionals.pop()
            
            if last_conditional or (last_conditional is None):
                conditionals.append(None)
            else:
                try:
                    conditionals.append(bool(_evaluate_statement(line[1], definitions)))
                except NameError:
                    #TODO: log warning
                    print("Warning: Evaluation of preprocessor directive failed.\n\t\"" + str(code[i]) + "\"")
                    conditionals.append(False)
        elif line.startswith("#else"):
            last_conditional = conditionals.pop()
            
            if last_conditional or (last_conditional is None):
                conditionals.append(None)
            else:
                conditionals.append(True)
        elif line.startswith("#endif"):
            conditionals.pop()

    return definitions

def _evaluate_statement(statement, definitions):
    statement = _resolve_identifiers(statement.strip(), definitions)
    
    if statement.startswith("defined("):
        statement = statement.replace("defined(", "")[:-1]
        return statement in definitions

    return eval(statement, {}, {})

def _resolve_identifiers(statement, definitions):
    keywords = {".eq.":"==", ".ne.":"!=", ".gt.":">", ".lt.":"<", ".ge.":">=", ".le.":"<=",
                ".and.":"and", ".or.":"or", ".not.":"not", ".eqv.":"==", ".neqv.":"!="}

    unresolved_tokens = False
    statement = _split_identifiers(statement)
    for i, token in enumerate(statement):
        if token in definitions:
            unresolved_tokens = True
            statement[i] = str(definitions[token])
    statement = "".join(statement)

    if unresolved_tokens:
        statement = _resolve_identifiers(statement, definitions)

    for keyword, value in keywords.items():
        statement = statement.replace(keyword, value)

    return statement
    
def _split_identifiers(string):
    i = 0
    output = []
    for j in range(len(string)):
        if not _isidentifier(string[i:j + 1]):
            output.append(string[i:j])
            i = j
    output.append(string[i:])
    return output

def _isidentifier(string):
    return bool(re.compile(r"^[a-zA-Z_]\w*$").match(string))
