/* glibc < 2.36 lacks arc4random; devkitA64's GCC only uses it for seeding. */
#include <stdint.h>
#include <sys/random.h>
uint32_t arc4random(void) {
    uint32_t v = 0;
    while (getrandom(&v, sizeof v, 0) != sizeof v) {}
    return v;
}
