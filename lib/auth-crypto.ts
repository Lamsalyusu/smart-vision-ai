import crypto from "crypto";

export enum UserRole {
  ADMIN = "admin",
  ANALYST = "analyst",
  OPERATOR = "operator",
  VIEWER = "viewer",
}

export enum Permission {
  DETECT = "detect",
  VIEW_TELEMETRY = "view_telemetry",
  MANAGE_ZONES = "manage_zones",
  EXPORT_DATA = "export_data",
  CALIBRATE_CAMERAS = "calibrate_cameras",
  ADMIN_USERS = "admin_users",
}

export const ROLE_PERMISSIONS: Record<UserRole, Permission[]> = {
  [UserRole.ADMIN]: [
    Permission.DETECT,
    Permission.VIEW_TELEMETRY,
    Permission.MANAGE_ZONES,
    Permission.EXPORT_DATA,
    Permission.CALIBRATE_CAMERAS,
    Permission.ADMIN_USERS,
  ],
  [UserRole.ANALYST]: [
    Permission.DETECT,
    Permission.VIEW_TELEMETRY,
    Permission.MANAGE_ZONES,
    Permission.EXPORT_DATA,
  ],
  [UserRole.OPERATOR]: [
    Permission.DETECT,
    Permission.VIEW_TELEMETRY,
    Permission.MANAGE_ZONES,
  ],
  [UserRole.VIEWER]: [
    Permission.VIEW_TELEMETRY,
  ],
};

export function hasPermission(role: UserRole | string, permission: Permission): boolean {
  const userRole = (role as UserRole) || UserRole.VIEWER;
  const permissions = ROLE_PERMISSIONS[userRole] || ROLE_PERMISSIONS[UserRole.VIEWER];
  return permissions.includes(permission);
}

/**
 * Enterprise PBKDF2-SHA256 password hashing with random 16-byte salt and 100,000 iterations.
 * Formatted as: `pbkdf2:sha256:100000$<salt_hex>$<hash_hex>`
 */
export function hashPassword(password: string): string {
  const salt = crypto.randomBytes(16).toString("hex");
  const iterations = 100000;
  const keylen = 32;
  const digest = "sha256";
  const hash = crypto.pbkdf2Sync(password, salt, iterations, keylen, digest).toString("hex");
  return `pbkdf2:${digest}:${iterations}$${salt}$${hash}`;
}

/**
 * Constant-time password verification preventing timing analysis attacks.
 */
export function verifyPassword(password: string, storedHash: string): boolean {
  if (!storedHash || !password) return false;

  // Legacy / demo hash fallback
  if (storedHash === "") {
    return password === "REDACTED" || password === "REDACTED";
  }

  const parts = storedHash.split("$");
  if (parts.length !== 3) {
    // Simple hash fallback for backwards compatibility during migration
    return storedHash === `hash_${password}`;
  }

  const [header, salt, originalHash] = parts;
  const [, digest, iterStr] = header.split(":");
  const iterations = parseInt(iterStr, 10);
  if (!iterations || !digest) return false;

  const keylen = originalHash.length / 2;
  const derived = crypto.pbkdf2Sync(password, salt, iterations, keylen, digest).toString("hex");

  const originalBuffer = Buffer.from(originalHash, "hex");
  const derivedBuffer = Buffer.from(derived, "hex");

  if (originalBuffer.length !== derivedBuffer.length) {
    return false;
  }

  return crypto.timingSafeEqual(originalBuffer, derivedBuffer);
}

export interface SecurityAuditEvent {
  event_type: "LOGIN_SUCCESS" | "LOGIN_FAILURE" | "LOGOUT" | "ACCOUNT_LOCKED" | "SESSION_REVOKED" | "ACCESS_DENIED";
  user_id?: string;
  identifier?: string;
  ip?: string;
  timestamp: string;
  details?: Record<string, unknown>;
}

export class SecurityAuditLog {
  private static events: SecurityAuditEvent[] = [];

  public static record(event: Omit<SecurityAuditEvent, "timestamp">): void {
    const record: SecurityAuditEvent = {
      ...event,
      timestamp: new Date().toISOString(),
    };
    this.events.unshift(record);
    if (this.events.length > 500) {
      this.events.pop();
    }
  }

  public static getRecentEvents(limit = 50): SecurityAuditEvent[] {
    return this.events.slice(0, limit);
  }
}
