import random

def generate_hex_ids(N, length=4):
    # Calculate the maximum value based on the length
    max_value = 16**length - 1
    
    # Check if N is greater than the number of possible unique values
    if N > max_value + 1:
        raise ValueError(f"Cannot generate {N} unique hexadecimal numbers with length {length}. Maximum possible is {max_value + 1}.")
    
    # Generate initial set of N random numbers
    initial_numbers = [random.randint(0, max_value) for _ in range(N)]
    unique_hex_numbers = {f"{num:0{length}X}" for num in initial_numbers}
    
    # Continue generating numbers until we have N unique values
    while len(unique_hex_numbers) < N:
        random_number = random.randint(0, max_value)
        hex_number = f"{random_number:0{length}X}"
        unique_hex_numbers.add(hex_number)
    
    return list(unique_hex_numbers)

# Example usage
N = 10
hex_numbers = generate_hex_ids(N)
print(hex_numbers)