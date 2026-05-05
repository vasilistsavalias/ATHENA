import { describe, expect, it } from "vitest";

import { mapBlockBChoiceKey, nextRouteFromProgress } from "@/lib/flow";

describe("nextRouteFromProgress", () => {
  it("routes to block A when block A has pending items", () => {
    expect(
      nextRouteFromProgress({
        block_a_completed: 3,
        block_a_total: 25,
        block_b_completed: 0,
        block_b_total: 15,
        block_c_completed: 0,
        block_c_total: 0,
        profile_completed: true,
        block_a_feedback_completed: false,
        block_b_feedback_completed: false,
        block_c_feedback_completed: true,
        is_complete: false,
      })
    ).toBe("/block-a");
  });

  it("routes to profile when session has no submitted items and profile is incomplete", () => {
    expect(
      nextRouteFromProgress({
        block_a_completed: 0,
        block_a_total: 25,
        block_b_completed: 0,
        block_b_total: 15,
        block_c_completed: 0,
        block_c_total: 0,
        profile_completed: false,
        block_a_feedback_completed: false,
        block_b_feedback_completed: false,
        block_c_feedback_completed: true,
        is_complete: false,
      })
    ).toBe("/profile");
  });

  it("skips block A entirely when block A total is zero", () => {
    expect(
      nextRouteFromProgress({
        block_a_completed: 0,
        block_a_total: 0,
        block_b_completed: 0,
        block_b_total: 20,
        block_c_completed: 0,
        block_c_total: 0,
        profile_completed: true,
        block_a_feedback_completed: true,
        block_b_feedback_completed: false,
        block_c_feedback_completed: true,
        is_complete: false,
      })
    ).toBe("/block-b");
  });

  it("routes to block A feedback after block A item completion", () => {
    expect(
      nextRouteFromProgress({
        block_a_completed: 25,
        block_a_total: 25,
        block_b_completed: 2,
        block_b_total: 15,
        block_c_completed: 0,
        block_c_total: 0,
        profile_completed: true,
        block_a_feedback_completed: false,
        block_b_feedback_completed: false,
        block_c_feedback_completed: true,
        is_complete: false,
      })
    ).toBe("/block-a-feedback");
  });

  it("routes to complete when done", () => {
    expect(
      nextRouteFromProgress({
        block_a_completed: 25,
        block_a_total: 25,
        block_b_completed: 15,
        block_b_total: 15,
        block_c_completed: 0,
        block_c_total: 0,
        profile_completed: true,
        block_a_feedback_completed: true,
        block_b_feedback_completed: true,
        block_c_feedback_completed: true,
        is_complete: true,
      })
    ).toBe("/complete");
  });

  it("routes to block C after block B is finished", () => {
    expect(
      nextRouteFromProgress({
        block_a_completed: 10,
        block_a_total: 10,
        block_b_completed: 17,
        block_b_total: 17,
        block_c_completed: 3,
        block_c_total: 16,
        profile_completed: true,
        block_a_feedback_completed: true,
        block_b_feedback_completed: false,
        block_c_feedback_completed: false,
        is_complete: false,
      })
    ).toBe("/block-b-feedback");
  });

  it("routes to block C feedback after block C items are finished", () => {
    expect(
      nextRouteFromProgress({
        block_a_completed: 10,
        block_a_total: 10,
        block_b_completed: 15,
        block_b_total: 15,
        block_c_completed: 10,
        block_c_total: 10,
        profile_completed: true,
        block_a_feedback_completed: true,
        block_b_feedback_completed: true,
        block_c_feedback_completed: false,
        is_complete: false,
      })
    ).toBe("/block-c-feedback");
  });
});

describe("mapBlockBChoiceKey", () => {
  it("maps choice keys", () => {
    expect(mapBlockBChoiceKey("a")).toBe("A");
    expect(mapBlockBChoiceKey("B")).toBe("B");
    expect(mapBlockBChoiceKey("t")).toBe("Tie");
    expect(mapBlockBChoiceKey("U")).toBe("Unsure");
    expect(mapBlockBChoiceKey("x")).toBeNull();
  });
});
