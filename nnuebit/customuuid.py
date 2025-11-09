import os
from typing import Self

UINT_128_MAX = (1 << 128) - 1
CLEAR_FLAG = ~0xf000_c000_0000_0000_0000
SET_FLAG   =  0x8000_8000_0000_0000_0000
EPOCH_BITS =       0x0fff_ff00_0000_0000
NEW_BITS   =              0xff_ffff_ffff

class UUID8:
    def __init__(self, bits: int | None = None, epoch: int | None = None) -> None:
        if (bits is not None) != (epoch is not None):
            raise
        if bits is None or epoch is None:
            self.bits = int.from_bytes(os.urandom(16))
            self.bits &= CLEAR_FLAG
            self.bits |= SET_FLAG
            self.bits &= ~EPOCH_BITS
            return

        if bits < 0 or bits > UINT_128_MAX:
            raise ValueError('uuid must be a 128 bit integer')

        if (bits & CLEAR_FLAG) | SET_FLAG != bits:
            raise ValueError('uuid is not a version 8 uuid')

        if epoch < 0 or epoch > 99_999:
            raise ValueError('epoch is out of range')

        if (bits & EPOCH_BITS) != (int(str(epoch), 16) << 40):
            raise ValueError('uuid and epoch do not match')

        self.bits = bits

    def set_epoch(self, epoch: int) -> Self:
        if epoch < 0 or epoch > 99_999:
            raise ValueError('epoch is out of range')
        epoch = int(str(epoch), 16)
        self.bits &= ~EPOCH_BITS
        self.bits |= epoch << 40
        self.bits &= ~NEW_BITS
        self.bits |= int.from_bytes(os.urandom(5))

        return self

    def __str__(self) -> str:
        return '%08x-%04x-%04x-%04x-%012x' % (self.bits >> 96, (self.bits >> 80) & 0xffff, (self.bits >> 64) & 0xffff, (self.bits >> 48) & 0xffff, self.bits & 0xffff_ffff_ffff)

    def __int__(self) -> int:
        return self.bits

if __name__ == '__main__':
    a = UUID8()
    print(a)
    print(a.set_epoch(99999))
    print(a.set_epoch(400))
    print(a.set_epoch(800))
    print(a.set_epoch(1200))
    b = a.set_epoch(12387)
    print(int(b))

    c = UUID8(int(b), 12387)
    print(c)
