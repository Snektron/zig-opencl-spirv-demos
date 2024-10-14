const builtin = @import("builtin");

pub const items_per_thread = 24;
pub const block_dim = 256;
pub const items_per_block = items_per_thread * block_dim;

var shared: [block_dim]f32 addrspace(.shared) = undefined;

fn syncThreads() void {
    asm volatile(
        \\OpControlBarrier %execution_scope %memory_scope %semantics
        :: [execution_scope] "" (@as(u32, 2)), // Workgroup scope
           [memory_scope] "" (@as(u32, 2)), // Workgroup scope
           [semantics] "" (@as(u32, 0x100 | 0x10)) // SequentiallyConsistent | WorkgroupMemory
    );
}

fn reduce(
    input: [*]const addrspace(.global) f32,
    output: [*]addrspace(.global) f32,
    last_block: u32,
    valid_in_last_block: u32,
) callconv(.Kernel) void {
    const bid = @workGroupId(0);
    const tid = @workItemId(0);
    const block_offset = bid * items_per_block;

    var total: f32 = 0;
    if (bid == last_block) {
        inline for (0..items_per_thread) |i| {
            const index = block_dim * i + tid;
            if (index < valid_in_last_block) {
                total += input[block_offset + block_dim * i + tid];
            }
        }
    } else {
        inline for (0..items_per_thread) |i| {
            total += input[block_offset + block_dim * i + tid];
        }
    }

    shared[tid] = total;

    syncThreads();

    comptime var i: usize = 1;
    inline while (i < block_dim) : (i <<= 1) {
        if (tid % (i * 2) == 0) {
            shared[tid] += (&shared)[tid + i];
        }
        syncThreads();
    }

    if (tid == 0) {
        output[bid] = (&shared)[0];
    }
}

comptime {
    // Only export the kernel when compiling for the device
    // so that if we @import this file from host it doesn't
    // try to reference the kernel.
    if (builtin.os.tag == .opencl) {
        @export(&reduce, .{ .name = "reduce" });
    }
}
