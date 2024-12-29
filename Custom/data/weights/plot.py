import pandas as pd
import matplotlib.pyplot as plt

###epoch

# Đọc dữ liệu từ file CSV
df = pd.read_csv('loss_values.csv')

# Tạo một DataFrame mới chỉ chứa thông tin của epoch cuối cùng cho mỗi phase
df_last_epoch = df.groupby(['phase', 'epoch']).tail(1)

# Plot biểu đồ
plt.figure(figsize=(12, 8))

# Plot local loss (loss_l)
plt.plot(df_last_epoch[df_last_epoch['phase'] == 'train']['epoch'], df_last_epoch[df_last_epoch['phase'] == 'train']['loc_loss'], label='Train Localization Loss')
plt.plot(df_last_epoch[df_last_epoch['phase'] == 'val']['epoch'], df_last_epoch[df_last_epoch['phase'] == 'val']['loc_loss'], label='Validation Localization Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Localization Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot classification loss (loss_c)
plt.figure(figsize=(12, 8))
plt.plot(df_last_epoch[df_last_epoch['phase'] == 'train']['epoch'], df_last_epoch[df_last_epoch['phase'] == 'train']['conf_loss'], label='Train Confidence Loss')
plt.plot(df_last_epoch[df_last_epoch['phase'] == 'val']['epoch'], df_last_epoch[df_last_epoch['phase'] == 'val']['conf_loss'], label='Validation Confidence Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classification Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot total loss (loss)
plt.figure(figsize=(12, 8))
plt.plot(df_last_epoch[df_last_epoch['phase'] == 'train']['epoch'], df_last_epoch[df_last_epoch['phase'] == 'train']['total_loss'], label='Train Total Loss')
plt.plot(df_last_epoch[df_last_epoch['phase'] == 'val']['epoch'], df_last_epoch[df_last_epoch['phase'] == 'val']['total_loss'], label='Validation Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Total Loss')
plt.legend()
plt.grid(True)
plt.show()

###iteration

# Đọc dữ liệu từ file CSV
df = pd.read_csv('loss_values.csv')

# Tạo một DataFrame mới chỉ chứa thông tin của epoch cuối cùng cho mỗi phase
df_last_iteration = df.groupby(['phase', 'iteration']).tail(1)

# Plot biểu đồ
plt.figure(figsize=(12, 8))

# Plot local loss (loss_l)
plt.plot(df_last_iteration[df_last_iteration['phase'] == 'train']['iteration'], df_last_iteration[df_last_iteration['phase'] == 'train']['loc_loss'], label='Train Localization Loss')
plt.plot(df_last_iteration[df_last_iteration['phase'] == 'val']['iteration'], df_last_iteration[df_last_iteration['phase'] == 'val']['loc_loss'], label='Validation Localization Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Localization Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot classification loss (loss_c)
plt.figure(figsize=(12, 8))
plt.plot(df_last_iteration[df_last_iteration['phase'] == 'train']['iteration'], df_last_iteration[df_last_iteration['phase'] == 'train']['conf_loss'], label='Train Confidence Loss')
plt.plot(df_last_iteration[df_last_iteration['phase'] == 'val']['iteration'], df_last_iteration[df_last_iteration['phase'] == 'val']['conf_loss'], label='Validation Confidence Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Classification Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot total loss (loss)
plt.figure(figsize=(12, 8))
plt.plot(df_last_iteration[df_last_iteration['phase'] == 'train']['iteration'], df_last_iteration[df_last_iteration['phase'] == 'train']['total_loss'], label='Train Total Loss')
plt.plot(df_last_iteration[df_last_iteration['phase'] == 'val']['iteration'], df_last_iteration[df_last_iteration['phase'] == 'val']['total_loss'], label='Validation Total Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Total Loss')
plt.legend()
plt.grid(True)
plt.show()
