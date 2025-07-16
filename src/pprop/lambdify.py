
def dict_to_lambdafunc(dictionary, jax = False):
    library = 'jnp' if jax else 'np'
    terms = []

    for key, p_value in dictionary.items():
        for value in p_value:
            factor = value[0]  # The factor, e.g., +1 or -1
            expr_terms = [f"{factor}"]  # Start with the factor

            for term in value[1:]:  # Process each sine/cosine term
                if term.startswith("s"):
                    index = int(term[1:])  # Extract index from 'sA'
                    expr_terms.append(f"{library}.sin(params[{index}])")
                elif term.startswith("c"):
                    index = int(term[1:])  # Extract index from 'cA'
                    expr_terms.append(f"{library}.cos(params[{index}])")

            # Join the terms for this expression (factor and trig functions)
            terms.append("*".join(expr_terms))

    # Join all the individual expressions with a '+' for the final function
    if len(terms) > 0:
        return " + ".join(terms)
    else:
        return "0"
