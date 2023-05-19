# Functions defining which epochs to log with the logging callbacks


def custom1(current_epoch: int):
    if current_epoch == 2 or current_epoch == 7:
        return True
    return False


def custom5000epochs(current_epoch: int):
    if current_epoch == 100:
        return True
    if current_epoch <= 4500:
        if current_epoch % 750 == 0:
            return True
    elif current_epoch <= 5000:
        return True
    return False