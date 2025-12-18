const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "nn-zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("nn.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Link Apple's Accelerate framework for AMX/BNNS
    exe.linkFramework("Accelerate");
    exe.linkLibC();

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the neural network engine");
    run_step.dependOn(&run_cmd.step);

    // Generate command (convenience)
    const gen_cmd = b.addRunArtifact(exe);
    gen_cmd.step.dependOn(b.getInstallStep());
    gen_cmd.addArgs(&.{ "--generate", "Once upon a time" });

    const gen_step = b.step("generate", "Generate text from trained model");
    gen_step.dependOn(&gen_cmd.step);
}
