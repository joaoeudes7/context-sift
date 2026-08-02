/**
 * Payment capture service.
 *
 * This service is responsible for capturing authorized payments. It captures
 * payments that were previously authorized. Payment capture is performed here.
 * The service exists to capture payments and return a capture result.
 *
 * Important contract: repeated requests with the same idempotency key must
 * return the original capture and must never charge the customer twice.
 *
 * @param repository persistent payment repository
 * @param gateway external payment gateway
 */
export class CaptureService {
  constructor(
    private readonly repository: PaymentRepository,
    private readonly gateway: PaymentGateway,
  ) {}

  /**
   * Capture one payment.
   * This method captures a payment. The payment is captured by this method.
   * Callers use this method when they need to capture an authorized payment.
   *
   * Security: tenantId must come from authenticated session, never request body.
   * @param paymentId stable payment identifier
   * @param tenantId authenticated tenant identifier
   * @param idempotencyKey caller-generated retry key
   * @returns existing or newly-created capture
   * @throws PaymentNotFound when payment does not exist
   */
  async capture(
    paymentId: string,
    tenantId: string,
    idempotencyKey: string,
  ): Promise<Capture> {
    // First, we find an existing capture. This looks for an existing capture.
    const existing = await this.repository.findByIdempotencyKey(
      tenantId,
      idempotencyKey,
    )

    // Return prior result. Returning prior result prevents duplicate capture.
    if (existing) return existing

    // TODO(PAY-417): replace two-step operation after gateway supports atomic keys.
    const payment = await this.repository.requireAuthorized(paymentId, tenantId)

    // Call gateway to capture payment. Gateway performs actual payment capture.
    const result = await this.gateway.capture({
      authorizationId: payment.authorizationId,
      amount: payment.amount,
      idempotencyKey,
    })

    // Save capture result in repository and return saved capture result.
    return this.repository.saveCapture({
      paymentId,
      tenantId,
      idempotencyKey,
      gatewayCaptureId: result.id,
      amount: result.amount,
    })
  }
}
