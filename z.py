import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import io

def parse_data(data_string):
    """
    Parses the checkpoint data from a string.

    Args:
        data_string (str): A string containing the data, with each
                           entry on a new line.

    Returns:
        tuple: A tuple containing two lists: (timesteps, losses)
    """
    timesteps = []
    losses = []
    # Use io.StringIO to treat the string as a file
    file = io.StringIO(data_string)
    for line in file:
        # Skip any empty or malformed lines
        if ':' not in line:
            continue
        
        # Split the line into two parts at the colon
        parts = line.strip().split(':')
        
        # The first part is "Checkpoint X", so we split it by space and take the second element
        timestep = int(parts[0].split()[1])
        
        # The second part is the loss value, which we convert to a float
        loss = float(parts[1])
        
        timesteps.append(timestep)
        losses.append(loss)
        
    return timesteps, losses

def plot_loss(timesteps, losses):
    """
    Plots the loss vs. timesteps.

    Args:
        timesteps (list): A list of integer timesteps.
        losses (list): A list of float loss values.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the data
    ax.plot(timesteps, losses, marker='o', linestyle='-', markersize=4, label='LoRA CPT (Ours)')
    # Make legend label text larger
    ax.legend(fontsize=24)

    # --- Add the reference point as requested ---
    ref_x, ref_y = 59000, 41.3
    ax.plot(ref_x, ref_y, marker='o', markersize=8, color='orange', linestyle='None', label='CPT (Reference)')
    ax.text(ref_x - 1000, ref_y, 'Reference ', color='orange', fontsize=17, ha='right', va='center')

    # --- Set plot labels and title ---
    ax.set_title('Pretraining Val. Loss Comparison', fontsize=22, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=18, fontweight='bold')
    ax.set_ylabel('Val. Perplexity', fontsize=18, fontweight='bold')

    # --- Set Y-axis to logarithmic scale ---
    ax.set_yscale('log')
    
    # --- Increase the font size of the ticks ---
    ax.tick_params(axis='both', which='major', labelsize=14)


    # --- Configure Ticks ---
    # To make the x-axis labels sparse, we use MaxNLocator, which automatically
    # finds up to 10 "nice" tick locations.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))
    
    # Format the x-tick labels to use a comma for thousands
    ax.get_xaxis().set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Ensure grid lines are visible and subtle
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Add a legend
    ax.legend(fontsize=20)
    
    # Improve layout
    plt.tight_layout()
    
    # Display the plot
    plt.savefig('loss_vs_timestep_updated.pdf')

# --- Main execution ---
if __name__ == "__main__":
    # 1. Read the data from the txt file
    with open('perplexities.txt', 'r') as f:
        data = f.read()

    # 2. Parse the data from the string
    checkpoint_timesteps, loss_values = parse_data(data)
    
    # 3. Plot the parsed data
    plot_loss(checkpoint_timesteps, loss_values)